'use client';
import { useRef, useState } from 'react';
import { IconCloudUpload, IconDownload, IconX } from '@tabler/icons-react';
import { Button, Group, Text, useMantineTheme, Notification } from '@mantine/core';
import { Dropzone } from '@mantine/dropzone';
import axios from 'axios';
import classes from './DropzoneButton.module.css';

export function DropzoneButton() {
  const theme = useMantineTheme();
  const openRef = useRef<() => void>(null);
  const [uploadStatus, setUploadStatus] = useState<'idle' | 'success' | 'error'>('idle');
  const [uploadMessage, setUploadMessage] = useState('');

  const handleDrop = async (files: File[]) => {
    if (files.length === 0) return;

    const file = files[0];
    const formData = new FormData();
    formData.append("image", file);

    try {
      const response = await axios.post(`${process.env.NEXT_PUBLIC_API_URL}/send-and-rec`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setUploadStatus('success');
      setUploadMessage(`Upload successful! File: ${response.data.filename}`);
      console.log("Upload Success:", response.data);
    } catch (error) {
      setUploadStatus('error');
      setUploadMessage('Upload failed. Please try again.');
      console.error("Upload Error:", error);
    }
  };

  return (
    <div className={classes.wrapper}>
      <Dropzone
        openRef={openRef}
        onDrop={handleDrop}
        className={classes.dropzone}
        radius="md"
        accept={["image/jpeg", "image/jpg"]}
        maxSize={30 * 1024 ** 2}
      >
        {/* ... rest of the Dropzone content ... */}
      </Dropzone>

      <Button className={classes.control} size="md" radius="xl" onClick={() => openRef.current?.()}>
        Select files
      </Button>

      {uploadStatus === 'success' && (
        <Notification color="teal" title="Success" onClose={() => setUploadStatus('idle')}>
          {uploadMessage}
        </Notification>
      )}

      {uploadStatus === 'error' && (
        <Notification color="red" title="Error" onClose={() => setUploadStatus('idle')}>
          {uploadMessage}
        </Notification>
      )}
    </div>
  );
}
