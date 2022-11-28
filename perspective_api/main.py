from googleapiclient import discovery
import numpy as np
import csv


def create_papi_client(api_key):
    return discovery.build("commentanalyzer",
                           "v1alpha1",
                           developerKey=api_key,
                           discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/"
                                               "rest?version=v1alpha1",
                           static_discovery=False)


def get_response_papi(client, att_list, sent):
    att_dict = dict.fromkeys(att_list)
    analyze_request = {'comment': {'text': sent},
                       'requestedAttributes': att_dict,
                       'languages': ['en']}
    response = client.comments().analyze(body=analyze_request).execute()
    return response


def extract_probs(response):
    response_array = np.array([])
    for att in sorted(response['attributeScores'].items()):
        prob = att[1]['spanScores'][0]['score']['value']
        response_array = np.append(response_array, prob)
    return response_array


def main():
    api_key = 'AIzaSyBxjpGxPUtN30WrjCYCC6pOPDZ4fhZpPRo'
    client = create_papi_client(api_key)

    att_list = ['FLIRTATION',
                'IDENTITY_ATTACK',
                'INSULT',
                'OBSCENE',
                'PROFANITY',
                'SEVERE_TOXICITY',
                'SEXUALLY_EXPLICIT',
                'THREAT',
                'TOXICITY']

    train_file = '../data/train.csv'

    with open(train_file, mode='r') as train:
        train_data = csv.DictReader(train)
        for row in train_data:
            response = get_response_papi(client, att_list, row['text'])
            probs = extract_probs(response)
            print(probs)


if __name__ == "__main__":
    main()
