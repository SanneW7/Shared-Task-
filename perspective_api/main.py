from googleapiclient import discovery
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


def pprint_response(response):
    for att in sorted(response['attributeScores'].items()):
        att_name = att[0]
        att_prob = att[1]['spanScores'][0]['score']['value']
        print('{0:<20}\t{1}'.format(att_name, att_prob))


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
            print(row['text'])
            print(row['label_sexist'])
            response = get_response_papi(client, att_list, row['text'])
            pprint_response(response)
            print()


if __name__ == "__main__":
    main()
