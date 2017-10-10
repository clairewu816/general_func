"""Use Google API to read / write data from Google Sheets
"""
import time
import httplib2
import shutil
from apiclient import discovery, errors
from oauth2client.file import Storage


class Gsheet(object):
    def __init__(self):
        self.credentials = self.get_credentials()
        self.service = self.build_service()

    def get_credentials(self):
        # Copy json file to tmp so that it can be re-writable
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

    def get_sheet_values(self, spreadsheet_id, range_name):
        try:
            result = self.service.spreadsheets().values().get(
                spreadsheetId=spreadsheet_id, range=range_name).execute()
        except errors.HttpError as err:
            print(err)
            print('First try failed and will try again in 60s.')
            time.sleep(60)
            result = self.service.spreadsheets().values().get(
                spreadsheetId=spreadsheet_id, range=range_name).execute()

        values = result.get('values', [])

        if not values:
            print('No data found.')
            return
        else:
            print('data fetched from google sheets')
            return values

    def write_sheet_values(self, spreadsheet_id, range_name, values, dimension):
        """
        More rules to format values can be found in:
        https://developers.google.com/sheets/api/samples/writing

        Args:
            dimension: 'ROWS' or 'COLUMNS'
        """
        body = {
            'values': values,
            'majorDimension': dimension
        }
        try:
            self.service.spreadsheets().values().update(
                valueInputOption='USER_ENTERED', spreadsheetId=spreadsheet_id, range=range_name,
                body=body).execute()
        except errors.HttpError as err:
            print(err)
            print('First try failed and will try again in 60s.')
            time.sleep(60)
            self.service.spreadsheets().values().update(
                valueInputOption='USER_ENTERED', spreadsheetId=spreadsheet_id, range=range_name,
                body=body).execute()
        return

    def get_sheet_id(self, spreadsheet_id, sheet_order):
        """Get sheet_id by the order in the spreadsheet
        """
        sheets = self.service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()['sheets']
        return sheets[sheet_order]['properties']['sheetId']

    def duplicate_sheet(self, spreadsheet_id, new_sheet_name):
        """Duplicate from the very first sheet and insert it as the very first one"""
        latest_sheet_id = self.get_sheet_id(spreadsheet_id, 0)

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
        request = self.service.spreadsheets().batchUpdate(spreadsheetId=spreadsheet_id, body=request_body)
        return request.execute()

    def clean_sheet(self, spreadsheet_id, sheet_id, end_column_index):
        """
        Clean values in the range of A2:end_column of sheetId 0 while keep format.

        Args:
            end_column_index (INT): zero index
        """
        request_body = {
            "requests": [
                {
                    "updateCells": {
                        "range": {
                            "sheetId": 0,
                            "startRowIndex": 1,
                            "startColumnIndex": 0,
                            "endColumnIndex": end_column_index
                        },
                        "fields": "userEnteredValue"
                    }
                }
            ]
        }
        request = self.service.spreadsheets().batchUpdate(spreadsheetId=spreadsheet_id, body=request_body)
        return request.execute()
