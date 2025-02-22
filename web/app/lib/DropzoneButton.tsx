'use client';
import { useRef } from 'react';
import { IconCloudUpload, IconDownload, IconX } from '@tabler/icons-react';
import { Button, Group, Text, useMantineTheme } from '@mantine/core';
import { Dropzone } from '@mantine/dropzone';
import classes from './DropzoneButton.module.css';

export function DropzoneButton() {
  const theme = useMantineTheme();
  const openRef = useRef<() => void>(null);

  // Function to handle file upload
  const handleDrop = async (files: File[]) => {
    if (files.length === 0) return;

    const file = files[0]; // Only take the first file
    const formData = new FormData();
    formData.append("image", file);

    try {
      const response = await fetch("http://localhost:8000/send-and-rec", {
        method: "POST",
        body: formData,
      });

      const result = await response.json();
      console.log("Upload Success:", result);
    } catch (error) {
      console.error("Upload Error:", error);
    }
  };

  return (
    <div className={classes.wrapper}>
      <Dropzone
        openRef={openRef}
        onDrop={handleDrop}  // Updated to send the file
        className={classes.dropzone}
        radius="md"
        accept={["image/jpeg"]}
        maxSize={30 * 1024 ** 2}
      >
        <div style={{ pointerEvents: 'none' }}>
          <Group justify="center">
            <Dropzone.Accept>
              <IconDownload size={50} color={theme.colors.blue[6]} stroke={1.5} />
            </Dropzone.Accept>
            <Dropzone.Reject>
              <IconX size={50} color={theme.colors.red[6]} stroke={1.5} />
            </Dropzone.Reject>
            <Dropzone.Idle>
              <IconCloudUpload size={50} stroke={1.5} />
            </Dropzone.Idle>
          </Group>

          <Text ta="center" fw={700} fz="lg" mt="xl">
            <Dropzone.Accept>Drop JPG files here</Dropzone.Accept>
            <Dropzone.Reject>Only JPG files under 30MB are allowed</Dropzone.Reject>
            <Dropzone.Idle>Upload an image</Dropzone.Idle>
          </Text>
          <Text ta="center" fz="sm" mt="xs" c="dimmed">
            Drag&apos;n&apos;drop JPG files here to upload. Only <i>.jpg</i> files under 30MB are accepted.
          </Text>
        </div>
      </Dropzone>

      <Button className={classes.control} size="md" radius="xl" onClick={() => openRef.current?.()}>
        Select files
      </Button>
    </div>
  );
}
