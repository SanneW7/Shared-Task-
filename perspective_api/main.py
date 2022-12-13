from googleapiclient import discovery
import csv
import json
import time


def create_papi_client(api_key):
    """Build the PerspectiveAPI client using an API key"""
    return discovery.build("commentanalyzer",
                           "v1alpha1",
                           developerKey=api_key,
                           discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/"
                                               "rest?version=v1alpha1",
                           static_discovery=False)


def get_response_papi(client, att_list, sent):
    """Get the attribute probabilities of a sentence"""
    att_dict = dict.fromkeys(att_list)
    analyze_request = {'comment': {'text': sent},
                       'requestedAttributes': att_dict,
                       'languages': ['en']}
    response = client.comments().analyze(body=analyze_request).execute()
    return response


def extract_probs(response):
    """Extract attribute probabilites from response"""
    response_array = []
    for att in sorted(response['attributeScores'].items()):
        prob = att[1]['spanScores'][0]['score']['value']
        response_array.append(prob)
    return response_array


def main():
    # Create PerspectiveAPI client
    api_key = 'AIzaSyBxjpGxPUtN30WrjCYCC6pOPDZ4fhZpPRo'
    client = create_papi_client(api_key)

    # List of used attributes
    att_list = ['FLIRTATION',
                'IDENTITY_ATTACK',
                'INSULT',
                'OBSCENE',
                'PROFANITY',
                'SEVERE_TOXICITY',
                'SEXUALLY_EXPLICIT',
                'THREAT',
                'TOXICITY']

    # Files to extract data from and store data in
    input_file = '../data/dev_task_c_entries.csv'
    output_file = '../data/papi_features_dev_task_c_entries.json'

    # Get attribute probabilities of all training data
    papi_dict = dict()
    with open(input_file, mode='r') as input_f:
        input_data = csv.DictReader(input_f)
        for row in input_data:
            response = get_response_papi(client, att_list, row['text'])
            probs = extract_probs(response)
            papi_dict[row['rewire_id']] = probs

            print('Retrieved and saved response of: {0}'.format(row['rewire_id']))
            time.sleep(1.1)

    # Save attribute probabilities to a JSON file
    with open(output_file, "w") as outfile:
        json.dump(papi_dict, outfile, indent=4)


if __name__ == "__main__":
    main()
