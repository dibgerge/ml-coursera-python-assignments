import json
import os
import pickle
from collections import OrderedDict

import numpy as np
import requests


class SubmissionBase:
    submit_url = 'https://www.coursera.org/api/onDemandProgrammingScriptSubmissions.v1?includes=evaluation'
    save_file = 'token.pkl'

    def __init__(self, assignment_slug, assignment_key, part_names, part_names_key):
        self.assignment_slug = assignment_slug
        self.assignment_key = assignment_key
        self.part_names = part_names
        self.part_names_key = part_names_key
        self.login = None
        self.token = None
        self.functions = OrderedDict()
        self.args = dict()

    def grade(self):
        print('\nSubmitting Solutions | Programming Exercise %s\n' % self.assignment_slug)
        self.login_prompt()

        # Evaluate the different parts of exercise
        parts = OrderedDict()
        for part_id, result in self:
            parts[self.part_names_key[part_id - 1]] = {'output': sprintf('%0.5f ', result)}
        response = self.request(parts)
        response = json.loads(response.decode("utf-8"))

        # if an error was returned, print it and stop
        if 'errorCode' in response:
            print(response['message'], response['details']['learnerMessage'])
            return

        # Print the grading table
        print('%43s | %9s | %-s' % ('Part Name', 'Score', 'Feedback'))
        print('%43s | %9s | %-s' % ('---------', '-----', '--------'))
        for index, part in enumerate(parts):
            part_feedback = response['linked']['onDemandProgrammingScriptEvaluations.v1'][0]['parts'][str(part)][
                'feedback']
            part_evaluation = response['linked']['onDemandProgrammingScriptEvaluations.v1'][0]['parts'][str(part)]
            score = '%d / %3d' % (part_evaluation['score'], part_evaluation['maxScore'])
            print('%43s | %9s | %-s' % (self.part_names[int(index) - 1], score, part_feedback))
        evaluation = response['linked']['onDemandProgrammingScriptEvaluations.v1'][0]
        total_score = '%d / %d' % (evaluation['score'], evaluation['maxScore'])
        print('                                  --------------------------------')
        print('%43s | %9s | %-s\n' % (' ', total_score, ' '))

    def login_prompt(self):
        if os.path.isfile(self.save_file):
            with open(self.save_file, 'rb') as f:
                login, token = pickle.load(f)
            reenter = input('Use token from last successful submission (%s)? (Y/n): ' % login)

            if reenter == '' or reenter[0] == 'Y' or reenter[0] == 'y':
                self.login, self.token = login, token
                return
            else:
                os.remove(self.save_file)

        self.login = input('Login (email address): ')
        self.token = input('Token: ')

        # Save the entered credentials
        if not os.path.isfile(self.save_file):
            with open(self.save_file, 'wb') as f:
                pickle.dump((self.login, self.token), f)

    def request(self, parts):
        payload = {
            'assignmentKey': self.assignment_key,
            'submitterEmail': self.login,
            'secret': self.token,
            'parts': dict(eval(str(parts)))}
        headers = {}

        r = requests.post(self.submit_url, data=json.dumps(payload), headers=headers)
        return r.content

    def __iter__(self):
        for part_id in self.functions:
            yield part_id

    def __setitem__(self, key, value):
        self.functions[key] = value


def sprintf(fmt, arg):
    """ Emulates (part of) Octave sprintf function. """
    if isinstance(arg, tuple):
        # for multiple return values, only use the first one
        arg = arg[0]

    if isinstance(arg, (np.ndarray, list)):
        # concatenates all elements, column by column
        return ' '.join(fmt % e for e in np.asarray(arg).ravel('F'))
    else:
        return fmt % arg
