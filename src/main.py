import json
from query import Query
from shakespeare_rag import ShakespeareRAG

rag_test_queries = {
    "factual_recall": {
        1: "Who says 'All the world's a stage'?",
        2: "In which play does the character Rosalind appear?",
        3: "What is the name of Hamlet's mother?",
        4: "Who kills Tybalt in Romeo and Juliet?",
        5: "What city is Othello set in at the beginning of the play?",
        6: "Which character speaks the line 'Out, damned spot'?",
        7: "Who are the three witches in Macbeth talking to in Act 1 Scene 3?",
        8: "What poison does Hamlet's uncle use to kill the king?",
        9: "Who is Malvolio's mistress in Twelfth Night?",
        10: "What item does Desdemona drop that becomes key evidence?"
    },

    "contextual_understanding": {
        11: "What happens right before Hamlet delivers 'To be, or not to be'?",
        12: "What event follows Caesar's assassination?",
        13: "What does the Porter scene accomplish in Macbeth?",
        14: "Who is standing with Lear when he says 'Blow, winds, and crack your cheeks'?",
        15: "What message does Friar Laurence send to Romeo that never arrives?",
        16: "Who overhears Beatrice and Benedick being teased about each other?",
        17: "What is the setting and situation of the 'Friends, Romans, countrymen' speech?",
        18: "What does Viola disguise herself as in Twelfth Night, and why?",
        19: "What warning does the soothsayer give Caesar before the Ides of March?",
        20: "What does Shylock demand as payment in The Merchant of Venice?"
    },

    "interpretive_thematic": {
        21: "What does Hamlet mean by 'Denmark's a prison'?",
        22: "How does Lady Macbeth manipulate her husband?",
        23: "What is the role of fate in Romeo and Juliet?",
        24: "How does disguise serve as a theme in As You Like It?",
        25: "Why does Othello trust Iago?",
        26: "What commentary does King Lear make on authority and madness?",
        27: "How is mercy contrasted with justice in The Merchant of Venice?",
        28: "Why is jealousy called a 'green-eyed monster'?",
        29: "What does Prospero's breaking of his staff symbolize in The Tempest?",
        30: "What moral lesson does Helena's pursuit in All's Well That Ends Well suggest?"
    },

    "comparative_reasoning": {
        31: "Compare Hamlet's hesitation to Macbeth's decisiveness.",
        32: "How do Shakespeare's fools differ between King Lear and Twelfth Night?",
        33: "What similarities exist between Portia in Merchant and Rosalind in As You Like It?",
        34: "Compare the father-daughter relationships in King Lear and Othello.",
        35: "How does the portrayal of ambition differ between Macbeth and Julius Caesar?",
        36: "Which characters are most driven by revenge across Shakespeare's plays?",
        37: "Compare the tragic flaws of Hamlet and Othello.",
        38: "How does Shakespeare use supernatural elements differently in Hamlet and Macbeth?",
        39: "What roles do women play in shaping the plots of Shakespeare's comedies?",
        40: "How does The Tempest serve as a reflection on Shakespeare's own career?"
    },

    "creative_synthesis": {
        41: "If you combined all of Shakespeare's villains into one, what qualities would dominate?",
        42: "Rewrite 'To be, or not to be' as if spoken by Lady Macbeth.",
        43: "What advice might Polonius give to Juliet?",
        44: "Summarize all of King Lear's decisions that lead to his downfall.",
        45: "Which play provides the clearest example of mistaken identity?",
        46: "Construct a timeline of events in Hamlet Act 1.",
        47: "Identify every reference to dreams across Shakespeare's plays.",
        48: "What references to fate or prophecy appear in Macbeth and Julius Caesar?",
        49: "Which characters experience redemption at the end of their plays?",
        50: "How does Shakespeare portray love differently in his comedies versus tragedies?"
    }
}

def main():
    rag = ShakespeareRAG()
    results = []
    for category, questions in rag_test_queries.items():
        print(f"Category: {category}")
        for q_num, question in questions.items():
            print(f"  Q{q_num}: {question}")
            query = Query(rag, question)
            result = query.run_query_pipeline()
            print(f"Query completed in {query.time_to_rank_results + query.time_to_generate_answers + query.time_to_grade_answers:.2f} seconds")
            results.append(result)
    
    # Save results to a jsonl file
    with open("results.jsonl", "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
