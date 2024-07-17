from __future__ import print_function

import mysql.connector
from mysql.connector import errorcode
from get_secrets import get_secret

mysql_secrets = get_secret("mysql_secrets")

config = {
    "user": mysql_secrets["username"],
    "password": mysql_secrets["password"],
    "host": mysql_secrets["host"],
    "port": mysql_secrets["port"],
}
DB_NAME = mysql_secrets["dbInstanceIdentifier"]

TABLES = {}
TABLES["installations"] = (
    "CREATE TABLE `installations` ("
    "  `app_id` VARCHAR(255),"
    "  `enterprise_id` VARCHAR(255),"
    "  `enterprise_name` VARCHAR(255),"
    "  `enterprise_url` VARCHAR(255),"
    "  `team_id` VARCHAR(255),"
    "  `team_name` VARCHAR(255),"
    "  `bot_token` VARCHAR(255),"
    "  `bot_id` VARCHAR(255),"
    "  `bot_user_id` VARCHAR(255),"
    "  `bot_scopes` TEXT,"
    "  `user_id` VARCHAR(255),"
    "  `user_token` VARCHAR(255),"
    "  `user_scopes` TEXT,"
    "  `incoming_webhook_url` VARCHAR(255),"
    "  `incoming_webhook_channel` VARCHAR(255),"
    "  `incoming_webhook_channel_id` VARCHAR(255),"
    "  `incoming_webhook_configuration_url` VARCHAR(255),"
    "  `is_enterprise_install` BOOLEAN,"
    "  `token_type` VARCHAR(255),"
    "  PRIMARY KEY (`team_id`)"
    ") ENGINE=InnoDB"
)

TABLES["states"] = (
    "CREATE TABLE `states` ("
    "  `state` VARCHAR(255),"
    "  `timestamp` DATETIME,"
    "  PRIMARY KEY (`state`)"
    ") ENGINE=InnoDB"
)

TABLES["consent"] = (
    "CREATE TABLE `consent` ("
    "  `team_id` VARCHAR(255),"
    "  `user_id` VARCHAR(255),"
    "  `timezone` VARCHAR(255),"
    "  PRIMARY KEY (`user_id`)"
    ") ENGINE=InnoDB"
)

TABLES["analysis_results"] = (
    "CREATE TABLE `analysis_results` ("
    "  `team_id` VARCHAR(255),"
    "  `team_size` VARCHAR(255),"
    "  `team_duration` VARCHAR(255),"
    "  `collaboration_type` VARCHAR(255),"
    "  `industry` VARCHAR(255),"
    "  `task_type` VARCHAR(255),"
    "  `timestamp` DATETIME,"
    "  `n_users_consented` DOUBLE,"
    "  `method` VARCHAR(255),"
    "  `result` JSON,"
    "  PRIMARY KEY (`team_id`)"
    ") ENGINE=InnoDB"
)

cnx = mysql.connector.connect(**config)
cursor = cnx.cursor()


def create_database(cursor):
    try:
        cursor.execute(
            "CREATE DATABASE {} DEFAULT CHARACTER SET 'utf8'".format(DB_NAME)
        )
    except mysql.connector.Error as err:
        print("Failed creating database: {}".format(err))
        exit(1)


# Function to create or reset tables
def create_or_reset_table(table_name, table_description):
    try:
        print("Creating table {}: ".format(table_name), end="")
        cursor.execute(table_description)
        print("OK")
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_TABLE_EXISTS_ERROR:
            # print(
            #     "Table {} already exists. Dropping and recreating...".format(table_name)
            # )
            # cursor.execute("DROP TABLE {}".format(table_name))
            # cursor.execute(table_description)
            # print("Table {} recreated.".format(table_name))
            print("Table already exists")
        else:
            print(err.msg)


try:
    cursor.execute("USE {}".format(DB_NAME))
except mysql.connector.Error as err:
    print("Database {} does not exists.".format(DB_NAME))
    if err.errno == errorcode.ER_BAD_DB_ERROR:
        create_database(cursor)
        print("Database {} created successfully.".format(DB_NAME))
        cnx.database = DB_NAME
    else:
        print(err)
        exit(1)

for table_name in TABLES:
    table_description = TABLES[table_name]
    create_or_reset_table(table_name, table_description)


cursor.close()
cnx.close()
