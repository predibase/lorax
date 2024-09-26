import requests
from pprint import pprint

# Define the URL
url = "http://localhost:8080/classify_batch"

# Define the test cases
test_cases = [
    {
        "input": {
            "inputs": ["Recurring Withdr SP CARPE HTTPSMYCARPE. NC"]
        },
        "expected_output": [
			  [
				{
				  "entity_group": "tx_acronym",
				  "score": 0.9937902,
				  "word": "recurring",
				  "start": 0,
				  "end": 9
				},
				{
				  "entity_group": "tx_acronym",
				  "score": 0.9914077,
				  "word": "withdr",
				  "start": 10,
				  "end": 16
				},
				{
				  "entity_group": "payment_platform",
				  "score": 0.9927614,
				  "word": "sp",
				  "start": 17,
				  "end": 19
				},
				{
				  "entity_group": "merchant",
				  "score": 0.58535314,
				  "word": "carpe",
				  "start": 20,
				  "end": 25
				},
				{
				  "entity_group": "url",
				  "score": 0.98667973,
				  "word": "httpsmycarpe.",
				  "start": 26,
				  "end": 39
				},
				{
				  "entity_group": "location",
				  "score": 0.96868145,
				  "word": "nc",
				  "start": 40,
				  "end": 42
				}
			  ]
			]
    }
]

# Function to compare entities with a tolerance for scores and return differences
def compare_entities(actual, expected, score_tolerance=1e-1):
    differences = []
    if len(actual) != len(expected):
        differences.append(f"Number of entities mismatch. Actual: {len(actual)}, Expected: {len(expected)}")
        return differences

    for i, (a, e) in enumerate(zip(actual, expected)):
        entity_key = 'entity' if 'entity' in a else 'entity_group'
        for key in [entity_key, 'word', 'start', 'end', 'index']:
            if a.get(key) != e.get(key):
                differences.append(f"Entity {i}, {key} mismatch. Actual: {a.get(key)}, Expected: {e.get(key)}")
        
        if abs(a['score'] - e['score']) > score_tolerance:
            differences.append(f"Entity {i}, score difference exceeds tolerance. Actual: {a['score']}, Expected: {e['score']}")

    return differences

# Function to run a single test case
def run_test_case(test_case):
    # Make the POST request
    response = requests.post(url, json=test_case['input'])

    # Parse the JSON response
    result = response.json()
    pprint(result)

    # Compare entities and get differences
    differences = compare_entities(result[0], test_case['expected_output'][0])

    if differences:
        print(f"Test failed for input: {test_case['input']['inputs']}")
        print("Differences found:")
        for diff in differences:
            print(f"  - {diff}")
        raise ValueError("Differences found")
    else:
        print(f"Test passed for input: {test_case['input']['inputs']}")

# Run all test cases
all_passed = True
for test_case in test_cases:
    try:
        run_test_case(test_case)
    except Exception as e:
        all_passed = False
        print(f"Error occurred during test: {str(e)}")

if all_passed:
    print("All tests passed successfully!")
else:
    print("Some tests failed. Please check the output above for details.")
