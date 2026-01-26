import os

class Config:
    SECRET_KEY = 'kunci-rahasia-anda-yang-sangat-aman-b9'
    MYSQL_HOST = 'localhost'
    MYSQL_USER = 'root'
    MYSQL_PASSWORD = ''
    MYSQL_DB = 'db_swipeer'
    UPLOAD_FOLDER = os.path.join('static', 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # Maksimal upload 16MB