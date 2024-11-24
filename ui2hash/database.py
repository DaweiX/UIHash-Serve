from os.path import join, abspath, dirname

import pymysql
import yaml
import numpy as np

from .util import init_logger


# load configs
home_path = dirname(dirname(abspath(__file__)))
with open(join(home_path, "config.yml"), 'r') as f:
    configs = yaml.safe_load(f)
DATABASE_HOST = configs["mysql"]["DATABASE_HOST"]
DATABASE_USER = configs["mysql"]["DATABASE_USER"]
DATABASE_PASSWORD = configs["mysql"]["DATABASE_PASSWORD"]
DATABASE_NAME = configs["mysql"]["DATABASE_NAME"]
LOG_FILE = configs["log"]["DATABASE"]
LOG_LEVEL = configs["log"]["LEVEL"]

TABLE_NAME = "ui_features"
KEY_APK = "apk_sha256"
KEY_UI = "ui_activity_name"
KEY_TYPE = "view_type"
KEY_REP = "uihash"

logger = init_logger(LOG_LEVEL, LOG_FILE, "UIDB")


def create_database():
    conn = pymysql.connect(host=DATABASE_HOST, user=DATABASE_USER, 
                           password=DATABASE_PASSWORD)
    cursor = conn.cursor()
    logger.info("Clean database")
    query = f"DROP DATABASE IF EXISTS {DATABASE_NAME}"
    cursor.execute(query)
    logger.info(f"Create database {DATABASE_NAME}")
    query = f"CREATE DATABASE IF NOT EXISTS {DATABASE_NAME}"
    cursor.execute(query)
    conn.commit()
    conn.close()
    cursor.close()
    logger.info("Database created")


def create_table():
    conn = pymysql.connect(host=DATABASE_HOST, user=DATABASE_USER, 
                           password=DATABASE_PASSWORD,
                           db=DATABASE_NAME)
    cursor = conn.cursor()
    query = f"""CREATE TABLE {TABLE_NAME} (
            id INT AUTO_INCREMENT PRIMARY KEY,
            {KEY_APK} VARCHAR(64) NOT NULL,
            {KEY_UI} TEXT NOT NULL,
            {KEY_TYPE} TEXT NOT NULL,
            {KEY_REP} BLOB NOT NULL,
            UNIQUE ({KEY_APK}, {KEY_UI}(255))
            )
           """
    cursor.execute(query)
    conn.commit()
    conn.close()
    cursor.close()
    logger.info("Database table initialized")


def insert_data(apk_sha256, ui_activity_name, type_str, nparray):
    nparray_blob = nparray.tobytes()
    
    conn = pymysql.connect(host=DATABASE_HOST, user=DATABASE_USER, 
                           password=DATABASE_PASSWORD,
                           db=DATABASE_NAME)
    cursor = conn.cursor()
    
    try:
        query = f"""
        REPLACE INTO {TABLE_NAME} ({KEY_APK}, {KEY_UI}, {KEY_TYPE}, {KEY_REP})
        VALUES (%s, %s, %s, %s)
        """
        cursor.execute(query, (apk_sha256, ui_activity_name, type_str, nparray_blob))
        conn.commit()
        logger.debug(f"Successfully add an item for: {apk_sha256}")
    except Exception as e:
        logger.error(f"Error inserting data: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()


def fetch_ui_data_by_apk(apk_sha256):
    conn = pymysql.connect(host=DATABASE_HOST, user=DATABASE_USER, 
                           password=DATABASE_PASSWORD,
                           db=DATABASE_NAME)
    cursor = conn.cursor()
    
    try:
        query = f"""
        SELECT {KEY_UI}, {KEY_REP} 
        FROM {TABLE_NAME} 
        WHERE {KEY_APK} = %s
        """
        cursor.execute(query, (apk_sha256,))
        results = cursor.fetchall()
        
        data = []
        for activity, nparray_blob in results:
            nparray = np.frombuffer(nparray_blob, dtype=np.float32).reshape(1, -1)
            data.append((activity, nparray))
        return data
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return []
    finally:
        cursor.close()
        conn.close()


def fetch_all_data():
    conn = pymysql.connect(host=DATABASE_HOST, user=DATABASE_USER, 
                           password=DATABASE_PASSWORD,
                           db=DATABASE_NAME)
    cursor = conn.cursor()
    
    try:
        query = f"""
        SELECT id, {KEY_APK}, {KEY_UI}, {KEY_REP} 
        FROM {TABLE_NAME}
        """
        cursor.execute(query)
        return cursor.fetchall()
    except Exception as e:
        logger.error(f"Error fetching all data: {e}")
        return []
    finally:
        cursor.close()
        conn.close()


def init_database():
    create_database()
    create_table()