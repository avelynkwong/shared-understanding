from slack_sdk.oauth.installation_store import FileInstallationStore
from slack_sdk.oauth.installation_store.models.installation import Installation
import mysql.connector
from dotenv import load_dotenv
import os

load_dotenv()

config = {
    "user": "root",
    "password": os.getenv("DB_PASSWORD"),
    "host": "localhost",
    "database": os.getenv("DB_NAME"),
}


class CustomFileInstallationStore(FileInstallationStore):

    def save(self, installation):

        insert_cmd = (
            "INSERT INTO installations ("
            "app_id, enterprise_id, enterprise_name, enterprise_url, team_id, team_name, "
            "bot_token, bot_id, bot_user_id, bot_scopes, user_id, user_token, user_scopes, "
            "incoming_webhook_url, incoming_webhook_channel, incoming_webhook_channel_id, "
            "incoming_webhook_configuration_url, is_enterprise_install, token_type"
            ") VALUES (%(app_id)s, %(enterprise_id)s, %(enterprise_name)s, %(enterprise_url)s, "
            "%(team_id)s, %(team_name)s, %(bot_token)s, %(bot_id)s, %(bot_user_id)s, %(bot_scopes)s, "
            "%(user_id)s, %(user_token)s, %(user_scopes)s, %(incoming_webhook_url)s, %(incoming_webhook_channel)s, "
            "%(incoming_webhook_channel_id)s, %(incoming_webhook_configuration_url)s, %(is_enterprise_install)s, %(token_type)s)"
        )

        try:
            # connect to db
            cnx = mysql.connector.connect(**config)
            cursor = cnx.cursor()
            # execute insert cmd
            cursor.execute(insert_cmd, installation)
            cnx.commit()
            print("Installation added to db successfully.")

        except mysql.connector.Error as err:
            print("Error: {}".format(err))

        finally:
            if cursor:
                cursor.close()
            if cnx:
                cnx.close()

    def find_installation(
        self,
        *,
        user_id: str | None = None,
        enterprise_id: str | None,
        team_id: str | None,
        is_enterprise_install: bool = None
    ):
        select_cmd = "SELECT * FROM installations WHERE team_id = %s"

        try:
            # connect to db
            cnx = mysql.connector.connect(**config)
            cursor = cnx.cursor(dictionary=True)
            # execute the select cmd
            cursor.execute(select_cmd, (team_id,))
            # fetch the result
            result = cursor.fetchone()
            if not result:
                print("Installation not found")
                return None
            else:
                print("Found installation")
                installation = Installation(
                    app_id=result["app_id"],
                    enterprise_id=result["enterprise_id"],
                    enterprise_name=result["enterprise_name"],
                    enterprise_url=result["enterprise_url"],
                    team_id=result["team_id"],
                    team_name=result["team_name"],
                    bot_token=result["bot_token"],
                    bot_id=result["bot_id"],
                    bot_user_id=result["bot_user_id"],
                    bot_scopes=result["bot_scopes"],
                    user_id=result["user_id"],
                    user_token=result["user_token"],
                    user_scopes=result["user_scopes"],
                    incoming_webhook_url=result["incoming_webhook_url"],
                    incoming_webhook_channel=result["incoming_webhook_channel"],
                    incoming_webhook_channel_id=result["incoming_webhook_channel_id"],
                    incoming_webhook_configuration_url=result[
                        "incoming_webhook_configuration_url"
                    ],
                    is_enterprise_install=result["is_enterprise_install"],
                    token_type=result["token_type"],
                )
                return installation

        except mysql.connector.Error as err:
            print("Error: {}".format(err))
            return False

        finally:
            if cursor:
                cursor.close()
            if cnx:
                cnx.close()

    def delete_installation(
        self,
        *,
        user_id: str | None = None,
        enterprise_id: str | None,
        team_id: str | None
    ) -> None:
        delete_cmd = "DELETE FROM installations WHERE team_id = %s"

        try:
            # connect to db
            cnx = mysql.connector.connect(**config)
            cursor = cnx.cursor()
            # execute delete cmd
            cursor.execute(delete_cmd, (team_id,))
            cnx.commit()
            # check if any row affected
            if cursor.rowcount == 0:
                print("No installations deleted")
            else:
                print("Installation deleted")

        except mysql.connector.Error as err:
            print("Error: {}".format(err))
            return False

        finally:
            if cursor:
                cursor.close()
            if cnx:
                cnx.close()
