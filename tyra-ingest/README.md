# Tyra Ingest Folder

This folder provides automatic file ingestion into the Tyra memory system. Simply drop files into the `inbox` folder and they will be automatically processed and stored in memory.

## Folder Structure

- **inbox/**: Drop files here for automatic processing
- **processed/**: Successfully processed files are moved here
- **failed/**: Files that failed processing are moved here for investigation

## Supported File Types

The file watcher supports the same file types as the existing document processor:
- `.pdf` - PDF documents
- `.docx` - Word documents
- `.txt` - Plain text files
- `.md` - Markdown files
- `.html` - HTML files
- `.json` - JSON files
- `.csv` - CSV files

## How It Works

1. **File Detection**: The file watcher monitors the `inbox` folder for new files
2. **Stability Check**: Files are checked for stability (no longer being written to)
3. **Processing**: Files are processed through the existing document ingestion pipeline
4. **Storage**: Content is embedded and stored in the memory system
5. **Organization**: Files are moved to `processed` or `failed` based on outcome

## Features

- **Duplicate Detection**: Files are checked for duplicates using MD5 hashing
- **Error Handling**: Failed files are moved to `failed` folder with error logs
- **Agent Assignment**: Files are assigned to the configured agent (default: "tyra")
- **Monitoring**: Processing statistics and health status are tracked
- **API Control**: Service can be controlled via REST API endpoints

## Configuration

File watcher settings can be configured in `config/config.yaml` under the `file_watcher` section:

```yaml
file_watcher:
  enabled: true
  processing:
    agent_id: "tyra"
    auto_start: true
    check_interval: 0.5
```

## API Endpoints

- `GET /v1/file-watcher/status` - Service status
- `GET /v1/file-watcher/stats` - Processing statistics
- `GET /v1/file-watcher/health` - Health check
- `POST /v1/file-watcher/start` - Start service
- `POST /v1/file-watcher/stop` - Stop service
- `POST /v1/file-watcher/restart` - Restart service

## Usage Example

1. Drop a PDF file into `inbox/`
2. File watcher detects the new file
3. Document processor extracts text and metadata
4. Content is embedded and stored in memory
5. File is moved to `processed/` folder
6. Content is now searchable via memory queries

## Monitoring

Check the service status and processing statistics:
```bash
curl http://localhost:8000/v1/file-watcher/status
curl http://localhost:8000/v1/file-watcher/stats
```

## Troubleshooting

If files are not being processed:
1. Check service status via API
2. Verify file watcher is enabled in config
3. Check logs for error messages
4. Ensure file types are supported
5. Verify folder permissions

Files that fail processing will be moved to the `failed/` folder with error information for investigation.