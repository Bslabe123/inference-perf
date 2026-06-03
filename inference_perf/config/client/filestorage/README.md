# Storage Configuration

Configuration for the `storage:` block, which controls where benchmark report
files are written. Reports can be persisted to a local directory, to Google
Cloud Storage (GCS), or to an S3 / S3-compatible object store. Multiple backends
can be enabled at once; local storage is always available.

This page is the field-level reference. For the high-level config overview and
full end-to-end examples, see [docs/config.md](../../../../docs/config.md).

Schema: [`config.py`](./config.py).

## Backends

| Key | Backend | Enabled by default |
| --- | --- | --- |
| `local_storage` | Local filesystem directory | Yes (always on) |
| `google_cloud_storage` | Google Cloud Storage bucket | No (`null` unless set) |
| `simple_storage_service` | AWS S3 or S3-compatible store | No (`null` unless set) |

All three backends share two base fields:

<!-- FIELDS: StorageConfigBase -->

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `path` | str | `reports-<timestamp>` | Destination directory (local) or key prefix (GCS / S3). The default is generated once per run from the start time, formatted reports-YYYYMMDD-HHMMSS. |
| `report_file_prefix` | str | `null` | Optional prefix prepended to each report filename. |

<!-- /FIELDS -->

## `local_storage`

Writes reports to a directory on the machine running the benchmark. This backend
is always active, so reports are written locally even when no other backend is
configured.

**When to use:** default for local runs and quick iteration; no external
credentials or buckets required.

`local_storage` takes the two base fields above (`path`, `report_file_prefix`).

```yaml
storage:
  local_storage:
    path: "reports-{timestamp}"
    report_file_prefix: null
```

## `google_cloud_storage`

Uploads reports to a GCS bucket. `bucket_name` is required; `path` acts as a key
prefix within the bucket.

**When to use:** runs on GCP, or when reports should be centralized in a GCS
bucket for sharing and retention.

<!-- FIELDS: GoogleCloudStorageConfig -->

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `path` | str | `reports-<timestamp>` | Destination directory (local) or key prefix (GCS / S3). The default is generated once per run from the start time, formatted reports-YYYYMMDD-HHMMSS. |
| `report_file_prefix` | str | `null` | Optional prefix prepended to each report filename. |
| `bucket_name` | str | (required) | Target GCS bucket. |

<!-- /FIELDS -->

```yaml
storage:
  google_cloud_storage:
    bucket_name: "your-bucket-name"
    path: "reports-{timestamp}"
    report_file_prefix: null
```

## `simple_storage_service`

Uploads reports to AWS S3 or any S3-compatible object store. `bucket_name` is
required; the remaining fields target custom endpoints and addressing schemes.

**When to use:** runs on AWS, or against an S3-compatible store (set
`endpoint_url`). Use `addressing_style: path` for stores that do not support
virtual-hosted-style buckets.

<!-- FIELDS: SimpleStorageServiceConfig -->

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `path` | str | `reports-<timestamp>` | Destination directory (local) or key prefix (GCS / S3). The default is generated once per run from the start time, formatted reports-YYYYMMDD-HHMMSS. |
| `report_file_prefix` | str | `null` | Optional prefix prepended to each report filename. |
| `bucket_name` | str | (required) | Target S3 bucket. |
| `endpoint_url` | str | `null` | Custom endpoint URL for S3-compatible stores. |
| `region_name` | str | `null` | AWS region name. |
| `addressing_style` | enum | `null` | Bucket addressing: auto, virtual, or path. |

<!-- /FIELDS -->

```yaml
storage:
  simple_storage_service:
    bucket_name: "your-bucket-name"
    path: "reports-{timestamp}"
    report_file_prefix: null
    endpoint_url: null
    region_name: null
    addressing_style: null
```
