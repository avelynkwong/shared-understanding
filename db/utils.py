import mysql.connector
from get_secrets import get_secret

mysql_secrets = get_secret("mysql_secrets")

config = {
    "user": mysql_secrets["username"],
    "password": mysql_secrets["password"],
    "host": mysql_secrets["host"],
    "port": mysql_secrets["port"],
    "database": mysql_secrets["dbInstanceIdentifier"],
}


def add_consent_record(team_id, user_id, tz, consented):
    insert_cmd = "INSERT INTO consent (team_id, user_id, timezone, consented) VALUES (%s, %s, %s, %s)"

    try:
        # connect to db
        cnx = mysql.connector.connect(**config)
        cursor = cnx.cursor()
        # execute insert cmd
        cursor.execute(insert_cmd, (team_id, user_id, tz, consented))
        cnx.commit()
        print(f"Consent record created for: {user_id}.")

    except mysql.connector.Error as err:
        print("Error: {}".format(err))

    finally:
        if cursor:
            cursor.close()
        if cnx:
            cnx.close()


def modify_consent(user_id, consented, tz=None):
    if consented:
        update_cmd = "UPDATE consent SET consented=%s, timezone=%s WHERE user_id=%s"
        try:
            # connect to db
            cnx = mysql.connector.connect(**config)
            cursor = cnx.cursor()
            # execute insert cmd
            cursor.execute(update_cmd, (consented, tz, user_id))
            cnx.commit()
            print(f"Consent record updated for: {user_id}.")

        except mysql.connector.Error as err:
            print("Error: {}".format(err))

        finally:
            if cursor:
                cursor.close()
            if cnx:
                cnx.close()
    else:
        update_cmd = "UPDATE consent SET consented=%s WHERE user_id=%s"
        try:
            # connect to db
            cnx = mysql.connector.connect(**config)
            cursor = cnx.cursor()
            # execute insert cmd
            cursor.execute(update_cmd, (consented, user_id))
            cnx.commit()
            print(f"Consent record updated for: {user_id}.")

        except mysql.connector.Error as err:
            print("Error: {}".format(err))

        finally:
            if cursor:
                cursor.close()
            if cnx:
                cnx.close()


def delete_user_consent(user_id):
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


def delete_team_consent(team_id):
    delete_cmd = "DELETE FROM consent WHERE team_id = %s"
    try:
        # connect to db
        cnx = mysql.connector.connect(**config)
        cursor = cnx.cursor()
        # execute delete cmd
        cursor.execute(delete_cmd, (team_id,))
        cnx.commit()
        # check if any row affected
        if cursor.rowcount == 0:
            print(f"No consent deleted for team {team_id}")
        else:
            print(f"Consent deleted for team {team_id}")

    except mysql.connector.Error as err:
        print("Error: {}".format(err))
        return False

    finally:
        if cursor:
            cursor.close()
        if cnx:
            cnx.close()


def get_received_form_users(team_id):
    select_cmd = "SELECT user_id FROM consent WHERE team_id = %s"
    try:
        # connect to db
        cnx = mysql.connector.connect(**config)
        cursor = cnx.cursor()
        # execute delete cmd
        cursor.execute(select_cmd, (team_id,))
        result = cursor.fetchall()
        received_forms = [row[0] for row in result]
        print("Received form users: ", received_forms)

    except mysql.connector.Error as err:
        print("Error: {}".format(err))
        return False

    finally:
        if cursor:
            cursor.close()
        if cnx:
            cnx.close()

    return received_forms


def get_consented_users(team_id):
    select_cmd = "SELECT user_id FROM consent WHERE team_id = %s AND consented=TRUE"
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


def add_reacts_db(team_id, timestamp, reacts_json):
    insert_cmd = (
        "INSERT INTO reacts (team_id, timestamp, react_data) VALUES (%s, %s, %s)"
    )

    try:
        # connect to db
        cnx = mysql.connector.connect(**config)
        cursor = cnx.cursor()
        # execute insert cmd
        cursor.execute(insert_cmd, (team_id, timestamp, reacts_json))
        cnx.commit()

    except mysql.connector.Error as err:
        print("Error: {}".format(err))

    finally:
        if cursor:
            cursor.close()
        if cnx:
            cnx.close()


def add_attachments_db(team_id, timestamp, attachment_json):
    insert_cmd = "INSERT INTO attachments (team_id, timestamp, attachment_data) VALUES (%s, %s, %s)"

    try:
        # connect to db
        cnx = mysql.connector.connect(**config)
        cursor = cnx.cursor()
        # execute insert cmd
        cursor.execute(insert_cmd, (team_id, timestamp, attachment_json))
        cnx.commit()

    except mysql.connector.Error as err:
        print("Error: {}".format(err))

    finally:
        if cursor:
            cursor.close()
        if cnx:
            cnx.close()


def add_analysis_db(
    team_id,
    leaders,
    team_size,
    team_duration,
    collaboration_type,
    industry,
    task_type,
    timestamp,
    n_users_consented,
    method,
    result,
):

    insert_cmd = "INSERT INTO analysis_results (team_id, leaders, team_size, team_duration, collaboration_type, industry, task_type, timestamp, n_users_consented, method, result) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"

    try:
        # connect to db
        cnx = mysql.connector.connect(**config)
        cursor = cnx.cursor()
        # execute insert cmd
        cursor.execute(
            insert_cmd,
            (
                team_id,
                leaders,
                team_size,
                team_duration,
                collaboration_type,
                industry,
                task_type,
                timestamp,
                n_users_consented,
                method,
                result,
            ),
        )
        cnx.commit()

    except mysql.connector.Error as err:
        print("Error: {}".format(err))

    finally:
        if cursor:
            cursor.close()
        if cnx:
            cnx.close()


def update_lsm_count_db(team_id, lsm_count, timestamp):

    update_cmd = "INSERT INTO lsm_count (team_id, timestamp, lsm_count) VALUES (%s, %s, %s) ON DUPLICATE KEY UPDATE timestamp = VALUES(timestamp), lsm_count = VALUES(lsm_count)"
    values = (team_id, timestamp, lsm_count)

    try:
        # connect to db
        cnx = mysql.connector.connect(**config)
        cursor = cnx.cursor()
        # Execute the SQL query
        cursor.execute(update_cmd, values)

        # Commit the transaction
        cnx.commit()

        print("Record inserted/updated successfully")

    except mysql.connector.Error as err:
        print("Error: {}".format(err))

    finally:
        if cursor:
            cursor.close()
        if cnx:
            cnx.close()


def get_prev_lsm_count(team_id):

    select_cmd = "SELECT timestamp, lsm_count FROM lsm_count WHERE team_id = %s"
    try:
        # connect to db
        cnx = mysql.connector.connect(**config)
        cursor = cnx.cursor()
        # execute insert cmd
        cursor.execute(select_cmd, (team_id,))
        record = cursor.fetchone()

        if not record:
            return None, None

        timestamp, lsm_count = record
        return timestamp, lsm_count

    except mysql.connector.Error as err:
        print("Error: {}".format(err))

    finally:
        if cursor:
            cursor.close()
        if cnx:
            cnx.close()
