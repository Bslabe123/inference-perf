# Copyright 2026 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class StorageConfigBase(BaseModel):
    path: str = Field(
        default=f"reports-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        description="Destination directory (local) or key prefix (GCS / S3). The default is generated once per run from the start time, formatted reports-YYYYMMDD-HHMMSS.",
    )
    report_file_prefix: Optional[str] = Field(default=None, description="Optional prefix prepended to each report filename.")


class GoogleCloudStorageConfig(StorageConfigBase):
    bucket_name: str = Field(..., description="Target GCS bucket.")


class SimpleStorageServiceConfig(StorageConfigBase):
    bucket_name: str = Field(..., description="Target S3 bucket.")
    endpoint_url: Optional[str] = Field(default=None, description="Custom endpoint URL for S3-compatible stores.")
    region_name: Optional[str] = Field(default=None, description="AWS region name.")
    addressing_style: Optional[Literal["auto", "virtual", "path"]] = Field(
        default=None, description="Bucket addressing: auto, virtual, or path."
    )


class StorageConfig(BaseModel):
    local_storage: StorageConfigBase = Field(
        default_factory=StorageConfigBase,
        description="Local filesystem directory backend (always on).",
    )
    google_cloud_storage: Optional[GoogleCloudStorageConfig] = Field(
        default=None, description="Google Cloud Storage bucket backend (null unless set)."
    )
    simple_storage_service: Optional[SimpleStorageServiceConfig] = Field(
        default=None, description="AWS S3 or S3-compatible store backend (null unless set)."
    )
