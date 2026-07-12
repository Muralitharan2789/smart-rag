EVAL_SET = [
    {
        "question": "What is the average rainfall in Asia in 2009 according to Table 9?",
        "expected_answer_contains": ["244"],
        "expected_source_document": "test_document.pdf",
    },
    {
        "question": "Who plays the character described as the Lovable ogre?",
        "expected_answer_contains": ["Robbie Coltrane"],
        "expected_source_document": "test_document.pdf",
    },
    {
        "question": "What is the capital of France?",
        "expected_answer_contains": ["don't", "cannot", "no information", "not", "no mention"],
        "expected_source_document": None,
    },
]