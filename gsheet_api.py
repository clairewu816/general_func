from __future__ import print_function
import httplib2
import os
import time

from apiclient import discovery, errors
from oauth2client.file import Storage

"""
Use Google API to read / write data from Google Sheets
"""


class Gsheet(object):
    def __init__(self):
        self.credentials = self.get_credentials()
        self.service = self.build_service()

    def get_credentials(self):
        credential_path = 'client_secret.json'

        store = Storage(credential_path)
        return store.get()

    def build_service(self):
        http = self.credentials.authorize(httplib2.Http())
        discoveryUrl = ('https://sheets.googleapis.com/$discovery/rest?'
                        'version=v4')
        return discovery.build('sheets', 'v4', http=http, discoveryServiceUrl=discoveryUrl)

    def get_sheet_values(self, spreadsheetId, rangeName):
        try:
            result = self.service.spreadsheets().values().get(
                spreadsheetId=spreadsheetId, range=rangeName).execute()
        except errors.HttpError as err:
            print(err)
            print('First try failed and will try again in 60s.')
            time.sleep(60)
            result = self.service.spreadsheets().values().get(
                spreadsheetId=spreadsheetId, range=rangeName).execute()

        values = result.get('values', [])

        if not values:
            print('No data found.')
            return
        else:
            print('data fetched from google sheets')
            return values

    def write_sheet_values(self, spreadsheetId, rangeName, values, dimension):
        """
        :param dimension: 'ROWS' or 'COLUMNS'
        # # https://developers.google.com/sheets/api/samples/writing
        """
        body = {
            'values': values,
            'majorDimension': dimension
        }
        try:
            self.service.spreadsheets().values().update(
                valueInputOption='USER_ENTERED', spreadsheetId=spreadsheetId, range=rangeName,
                body=body).execute()
        except errors.HttpError as err:
            print(err)
            print('First try failed and will try again in 60s.')
            time.sleep(60)
            self.service.spreadsheets().values().update(
                valueInputOption='USER_ENTERED', spreadsheetId=spreadsheetId, range=rangeName,
                body=body).execute()
        return
