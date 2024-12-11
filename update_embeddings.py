import os
import json
import psycopg2
from google.oauth2 import service_account
from googleapiclient.discovery import build
from sentence_transformers import SentenceTransformer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database connection parameters
db_params = {
    'host': os.getenv('SUPABASE_HOST'),
    'dbname': 'postgres',
    'user': 'postgres.fhoyagohhletichibcgm',  # Your Supabase user
    'password': os.getenv('SUPABASE_PASSWORD'),
    'port': '6543',
    'sslmode': 'require'
}

def get_drive_service():
    """Initialize Google Drive service"""
    try:
        credentials = service_account.Credentials.from_service_account_file(
            'service_account.json',
            scopes=['https://www.googleapis.com/auth/drive.readonly']
        )
        return build('drive', 'v3', credentials=credentials)
    except Exception as e:
        logger.error(f"Error initializing Drive service: {e}")
        raise

def read_file_content(service, file_id):
    """Read content from a Google Drive file"""
    try:
        content = service.files().get_media(fileId=file_id).execute()
        return content.decode('utf-8')
    except Exception as e:
        logger.error(f"Error reading file {file_id}: {e}")
        return None

def get_processed_files(conn):
    """Get list of already processed files"""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT DISTINCT metadata->>'file_id' 
            FROM documents 
            WHERE metadata->>'file_id' IS NOT NULL
        """)
        return {row[0] for row in cur.fetchall()}

def main():
    try:
        # Initialize services
        drive_service = get_drive_service()
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Connect to database
        conn = psycopg2.connect(**db_params)
        
        # Get list of processed files
        processed_files = get_processed_files(conn)
        
        # List all text files in Drive
        results = drive_service.files().list(
            q="mimeType='text/plain'",
            fields="files(id, name, modifiedTime)"
        ).execute()
        
        files = results.get('files', [])
        logger.info(f"Found {len(files)} text files in Drive")
        
        # Process new files
        new_files_count = 0
        for file in files:
            if file['id'] not in processed_files:
                content = read_file_content(drive_service, file['id'])
                if content:
                    # Generate embedding
                    embedding = model.encode(content)
                    
                    # Store in database
                    with conn.cursor() as cur:
                        cur.execute("""
                            INSERT INTO documents (content, metadata, embedding)
                            VALUES (%s, %s, %s)
                        """, (
                            content,
                            json.dumps({
                                "file_id": file['id'],
                                "file_name": file['name'],
                                "modified_time": file['modifiedTime']
                            }),
                            embedding.tolist()
                        ))
                    conn.commit()
                    new_files_count += 1
                    logger.info(f"Processed new file: {file['name']}")
        
        logger.info(f"Completed processing {new_files_count} new files")
        
    except Exception as e:
        logger.error(f"Error in main process: {e}")
        raise
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    main()
