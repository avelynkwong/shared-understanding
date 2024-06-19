from slack_sdk.oauth.state_store import FileOAuthStateStore
from uuid import uuid4
import time
from dotenv import load_dotenv
import os
import mysql.connector

load_dotenv()

config = {
    "user": "root",
    "password": os.getenv("DB_PASSWORD"),
    "host": "localhost",
    "database": os.getenv("DB_NAME"),
}


class CustomFileOAuthStateStore(FileOAuthStateStore):
    def __init__(self, *, expiration_seconds: int):
        self.expiration_seconds = expiration_seconds

    def issue(self, *args, **kwargs) -> str:
        state = str(uuid4())
        ts = time.time()
        insert_cmd = "INSERT INTO states (state, timestamp) VALUES (%s, %s)"

        try:
            # connect to db
            cnx = mysql.connector.connect(**config)
            cursor = cnx.cursor()
            # execute insert cmd
            cursor.execute(insert_cmd, (state, ts))
            cnx.commit()
            print("State added to db successfully.")

        except mysql.connector.Error as err:
            print("Error: {}".format(err))

        finally:
            if cursor:
                cursor.close()
            if cnx:
                cnx.close()

        return state

    def consume(self, state: str) -> bool:

        select_cmd = "SELECT * from states WHERE state = %s"
        delete_cmd = "DELETE FROM states WHERE state = %s"
        try:
            # connect to db
            cnx = mysql.connector.connect(**config)
            cursor = cnx.cursor(dictionary=True)
            # execute the select cmd
            cursor.execute(select_cmd, (state,))
            # fetch the result
            result = cursor.fetchone()
            if not result:
                print("Failed to find any persistent data for state: {state} - {e}")
                return False
            else:
                now = time.time()
                created = float(result["timestamp"])
                expiration = created + self.expiration_seconds
                print(expiration - now)
                still_valid: bool = now < expiration
                # delete from table
                cursor.execute(delete_cmd, (state,))
                cnx.commit()
                if cursor.rowcount == 0:
                    print("State deletion unsuccessful.")
                else:
                    print("State deletion successful.")
                return still_valid

        except mysql.connector.Error as err:
            print("Error: {}".format(err))
            return False

        finally:
            if cursor:
                cursor.close()
            if cnx:
                cnx.close()
