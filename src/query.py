import json
from pathlib import Path
from smolagents import OpenAIServerModel
from shakespeare_rag import ShakespeareRAG
import time


class Query:
    def __init__(self, rag: ShakespeareRAG, question: str, base_url: str = "http://localhost:1234/v1", model_name: str = "openai/gpt-oss-20b"):
        self.rag = rag
        self.question = question
        self.filters = None
        self.base_url = base_url
        self.model_name = model_name
        self.llm = OpenAIServerModel(
            model_id=model_name,
            api_base=base_url,
            api_key="dummy"
        )
        self.results = []
        self.answers = []
        self.grades = []
        self.time_to_rank_results = None
        self.time_to_generate_answers = None
        self.time_to_grade_answers = None
    
    def extract_filters(self):
        system_prompt = f"""
You are an information extraction model that can extract metadata from questions about Shakespeare's plays.
The only metadata you can extract are: play name, act number, scene number, and speaker name.
DO NOT answer the question. Only return metadata as JSON.

Schema:
{{
  "play": string | null,
  "act": integer | null,
  "scene": integer | null,
  "speaker": string | null
}}

Return only JSON.
"""

        response = self.llm.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": self.question}],
            max_tokens=200,
            temperature=0.0
        )
        try:
            content = response.choices[0].message.content
            #print(f"LLM Extracted Content: {content}")
            metadata = json.loads(content)
            # Clean up metadata: convert act and scene to int if possible
            if metadata.get("act") is not None:
                try:
                    metadata["act"] = int(metadata["act"])
                except:
                    metadata["act"] = None
            if metadata.get("scene") is not None:
                try:
                    metadata["scene"] = int(metadata["scene"])
                except:
                    metadata["scene"] = None
            self.filters = {k: v for k, v in metadata.items() if v is not None}
            return self.filters
        except json.JSONDecodeError:
            self.filters = {}
            return self.filters

    def rank_results(self, n_results: int = 5):
        start_time = time.time()
        results = self.rag.query_shakespeare(self.question, n_results=n_results)
        self.results = []
        for i, (doc, metadata, distance) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )):
            self.results.append({
                "rank": i + 1,
                "document": doc,
                "metadata": metadata,
                "relevance_score": 1 - distance
            })
        self.time_to_rank_results = time.time() - start_time
        return self.results
    
    def generate_answers(self):
        start_time = time.time()
        system_prompt = "You are a Shakespeare research assistant. You must answer questions about Shakespeare using only the reference passages provided. Do not use any other knowledge you may have about Shakespeare, his works, or his life. If the answer is not in the references, reply exactly with: 'The answer is not contained in the provided passages.'"
        self.answers = []
        for result in self.results:
            prompt = self._create_prompt(result)
            answer = self.llm.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.0
            ).choices[0].message.content
            self.answers.append({
                "rag_result": result,
                "rag_rank": result["rank"],
                "prompt": prompt,
                "answer": answer,
            })
        self.time_to_generate_answers = time.time() - start_time
        return self.answers
    
    def _create_prompt(self, result: dict) -> str:
        """Example result dict:
        {'rank': 1, 'document': "[Context before: ROSALIND: Say 'a day,' without the 'ever.' No, no, Orlando; men are April when they woo, December when they wed: maids are May when they are maids, but the sky changes when they are wives. I will be more jealous of thee than a Barbary cock-pigeon over his hen, more clamorous than a parrot against rain, more new-fangled than an ape, more giddy in my desires than a monkey: I will weep for nothing, like Diana in the fountain, and I will do that when you are disposed to be merry; I will laugh like a hyen, and that when thou art inclined to sleep.] But will my Rosalind do so? [Context after: ROSALIND: By my life, she will do as I do.]", 'metadata': {'act': 'IV', 'speaker': 'ORLANDO', 'scene': 'I', 'line_end': '4.1.140', 'play': 'As You Like It', 'line_start': '4.1.140'}, 'relevance_score': 0.6880857944488525}, 
        """
        document = result["document"]
        # the document is in the format [Context before: ...] passage... [Context after: ...]
        context_before = document.split("[Context before: ")[1].split("]")[0]
        context_after = document.split("[Context after: ")[1].split("]")[0]
        passage = document.split("] ")[1].split(" [Context after:")[0]
        #print(f"Context Before: {context_before}")
        #print(f"Context After: {context_after}")
        #print(f"Passage: {passage}")
        prompt = f"""
Context before: {context_before}
Passage: {passage}
Context after: {context_after}
Play: {result['metadata']['play']}
Act: {result['metadata']['act']}, Scene: {result['metadata']['scene']}
Speaker: {result['metadata']['speaker']}

Question: {self.question}

Task:
Answer using only the context above. Cite the passages you used. 
If the context is insufficient, respond: "The answer is not contained in the provided passages."
"""
        #print(f"Generated Prompt: {prompt}")
        return prompt
    
    def grade_answers(self):
        start_time = time.time()
        system_prompt = """
You are a grading assistant for a Shakespeare Retrieval-Augmented Generation (RAG) system.  
Your purpose is to evaluate whether a model's answer follows the task instructions and uses 
only the retrieved passages as evidence.  

You must:
- Judge based solely on the provided data (RAG result, prompt, and answer).
- Be objective, concise, and consistent.
- Do not rewrite or fix the answer — only grade it.

Use the following rubric to assign scores and explain your reasoning.
"""
        self.grades = []
        for answer_obj in self.answers:
            grade_prompt = self._create_judge_prompt(answer_obj)
            grade = self.llm.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": grade_prompt}],
                max_tokens=500,
                temperature=0.0
            ).choices[0].message.content
            self.grades.append({
                "rank": answer_obj["rag_rank"],
                "answer": answer_obj["answer"],
                "grade": grade,
                "relevance_score": answer_obj["rag_result"]["relevance_score"]
            })
        self.time_to_grade_answers = time.time() - start_time
        return self.grades
    
    def _create_judge_prompt(self, answer_obj: dict) -> str:
        prompt = f"""
### INPUT DATA
RAG Result:
{answer_obj['rag_result']['document']}

Prompt (the one used for generation):
{answer_obj['prompt']}

Model's Answer:
{answer_obj['answer']}

### GRADING CRITERIA

1. **Faithfulness to Context (0-3 points)**
   - 3 = Uses only information from the provided passage(s).
   - 2 = Mostly relies on the passage but includes mild inference.
   - 1 = Includes clear external knowledge or speculation.
   - 0 = Contradicts or ignores the passage entirely.

2. **Accuracy (0-3 points)**
   - 3 = Fully correct given the passage.
   - 2 = Mostly correct but missing or misphrasing a detail.
   - 1 = Partly correct or unclear.
   - 0 = Incorrect.

3. **Instruction Compliance (0-2 points)**
   - 2 = Fully follows the task (e.g., says “The answer is not contained…” when appropriate).
   - 1 = Minor deviation.
   - 0 = Violates or ignores instructions.

4. **Citation or Reference Use (0-2 points)**
   - 2 = Correctly cites or refers to the passage(s).
   - 1 = Vague or incomplete citation.
   - 0 = No or irrelevant citation.

### OUTPUT FORMAT

Respond in **JSON** with the following structure:

{{
  "faithfulness_score": <0-3>,
  "accuracy_score": <0-3>,
  "instruction_compliance_score": <0-2>,
  "citation_score": <0-2>,
  "total_score": <sum>,
  "explanation": "<concise justification of grading>",
  "final_verdict": "<PASS or FAIL>",
  "verdict_reason": "<why the response passes or fails>"
}}

A total score of 8 or higher counts as **PASS**.

Now evaluate the model's response according to the rubric.

Question: {self.question}

Retrieved Context:
{answer_obj['rag_result']['document']}

Generated Answer:
{answer_obj['answer']}

Evaluate this answer on a scale of 1-5 based on:
- Relevance: Does it answer the question?
- Accuracy: Is the information correct based on the context?
- Completeness: Is the answer thorough?

Provide your evaluation in the following format:
Score: [1-5]
Reasoning: [brief explanation]"""
        return prompt
    
    def run_query_pipeline(self, n_results: int = 5):
        self.rank_results(n_results)
        self.generate_answers()
        self.grade_answers()
        return {
            "question": self.question,
            "results": self.results,
            "answers": self.answers,
            "grades": self.grades,
            "time_to_rank_results": self.time_to_rank_results,
            "time_to_generate_answers": self.time_to_generate_answers,
            "time_to_grade_answers": self.time_to_grade_answers
        }