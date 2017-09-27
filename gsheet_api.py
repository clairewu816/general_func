from __future__ import print_function
import httplib2
import os
import time
import shutil
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
        """
        Copy json file to tmp so that it can be re-writable
        """
        shutil.copy2('client_secret.json', '/tmp')
        credential_path = '/tmp/truck-availability-googleapis.json'

        store = Storage(credential_path)
        credentials = store.get()
        return credentials

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

    def duplicate_sheet(self, spreadsheetId, new_sheet_name):
        """Duplicate last business day's sheet (aka the very first sheet) as the sheet for today"""

        # get latest sheet id
        sheets = self.service.spreadsheets().get(spreadsheetId=spreadsheetId).execute()['sheets']
        latest_sheet_id = sheets[0]['properties']['sheetId']

        # duplicate sheet
        request_body = {
            "requests": [
                {
                    "duplicateSheet": {
                        "sourceSheetId": latest_sheet_id, 
                        "insertSheetIndex": 0, 
                        "newSheetName": new_sheet_name
                    }
                }
            ]
        }
        request = self.service.spreadsheets().batchUpdate(spreadsheetId=spreadsheetId, body=request_body)
        response = request.execute()
        return
    
    def clean_sheet(self, spreadsheetId, end_column_index):
        """
        Clean values in the range of A2:end_column of sheetId 0 while keep format.
        :param end_column_index: zero index
        """
        request_body = {
            "requests": [
                {
                    "updateCells": {
                        "range": {
                            "sheetId": 0,
                            "startRowIndex": 1,
                            "startColumnIndex": 0,
                            "endColumnIndex": end_column_index - 1
                        },
                        "fields": "userEnteredValue"
                    }
                }
            ]
        }
        request = self.service.spreadsheets().batchUpdate(spreadsheetId=spreadsheetId, body=request_body)
        response = request.execute()
        return
    
