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


def add_consent_db(team_id, user_id):
    insert_cmd = "INSERT INTO consent (team_id, user_id) VALUES (%s, %s)"

    try:
        # connect to db
        cnx = mysql.connector.connect(**config)
        cursor = cnx.cursor()
        # execute insert cmd
        cursor.execute(insert_cmd, (team_id, user_id))
        cnx.commit()
        print(f"Consent recorded in DB for user {user_id}.")

    except mysql.connector.Error as err:
        print("Error: {}".format(err))

    finally:
        if cursor:
            cursor.close()
        if cnx:
            cnx.close()


def delete_consent_db(user_id):
    delete_cmd = "DELETE FROM consent WHERE user_id = %s"
    try:
        # connect to db
        cnx = mysql.connector.connect(**config)
        cursor = cnx.cursor()
        # execute delete cmd
        cursor.execute(delete_cmd, (user_id,))
        cnx.commit()
        # check if any row affected
        if cursor.rowcount == 0:
            print(f"No consent revoked in DB for {user_id}")
        else:
            print(f"Consent revoked in DB for {user_id}")

    except mysql.connector.Error as err:
        print("Error: {}".format(err))
        return False

    finally:
        if cursor:
            cursor.close()
        if cnx:
            cnx.close()


def get_consented_users(team_id):
    select_cmd = "SELECT user_id FROM consent WHERE team_id = %s"
    try:
        # connect to db
        cnx = mysql.connector.connect(**config)
        cursor = cnx.cursor()
        # execute delete cmd
        cursor.execute(select_cmd, (team_id,))
        result = cursor.fetchall()
        consented_users = [row[0] for row in result]
        print("Consented Users: ", consented_users)

    except mysql.connector.Error as err:
        print("Error: {}".format(err))
        return False

    finally:
        if cursor:
            cursor.close()
        if cnx:
            cnx.close()

    return consented_users
