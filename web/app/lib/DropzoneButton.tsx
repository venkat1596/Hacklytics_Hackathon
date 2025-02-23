'use client';
import { useRef, useState, useEffect } from 'react';
import { IconCloudUpload, IconDownload, IconX } from '@tabler/icons-react';
import { Button, Group, Text, useMantineTheme, RingProgress, Grid, Center, Box, Image } from '@mantine/core';
import { Dropzone } from '@mantine/dropzone';
import classes from './DropzoneButton.module.css';
import axios from 'axios';

export function DropzoneButton() {
  const theme = useMantineTheme();
  const openRef = useRef<() => void>(null);
  const [uploadStatus, setUploadStatus] = useState<'idle' | 'success' | 'error'>('idle');
  const [uploadMessage, setUploadMessage] = useState('');
  const [upScaled, setUpscaled] = useState<any>(null);
  const [loading, changelstate] = useState<number>(0);
  const [show, ChangeShow] = useState<number>(1);
  const [inputImage, setInputImage] = useState<string | null>(null); // State to store the input image

  // Function to handle file upload
  const handleDrop = async (files: File[]) => {
    console.log("Within Function");
    if (!files.length) return;
    while (loading);
    changelstate(1);
    const file = files[0];
    const data = new FormData();
    data.append('image', file, file.name); // Wrap the file in FormData

    // Set the input image for display
    const inputImageUrl = URL.createObjectURL(file);
    setInputImage(inputImageUrl);

    try {
      const response = await axios.post("http://127.0.0.1:8000/send-and-rec", data, {
        headers: {
          'accept': 'application/json',
          'Accept-Language': 'en-US,en;q=0.8',
          'Content-Type': 'multipart/form-data',
        },
        responseType: 'blob'
      });
      const blob = new Blob([response.data], { type: 'image/jpeg' });
      const imageUrl = URL.createObjectURL(blob);
      setUpscaled(imageUrl);
      console.log(imageUrl);
      console.log('Upload success:', response.data);
    } catch (error) {
      console.error('Upload failed:', error);
    }
    changelstate(0);
  };

  return (
    <div className={classes.wrapper}>
      <Dropzone
        openRef={openRef}
        onDrop={(files) => handleDrop(files)}  // Updated to send the file
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

      {/* Centered Lazy Loader */}
      <Center mt="md">
        {loading && (
          <RingProgress
            sections={[
              { value: 60, color: 'cyan' },
              { value: 20, color: 'pink' }
            ]}
          />
        )}
      </Center>

      {/* 2-Grid Display for Input and Output Images */}
{!loading && (inputImage || upScaled) && (
  <Grid mt="md" gutter="md">
    <Grid.Col span={6}>
      <Box style={{ border: '1px solid #ddd', padding: '8px', borderRadius: '8px' }}>
        <Text ta="center" fw={500} mb="sm">Input Image</Text>
        {inputImage && (
          <Image 
            src={inputImage} 
            alt="Input" 
            fit="contain" 
            width="50%" // Set the width to 50% to cut the size in half
            height="auto" // Maintain aspect ratio
          />
        )}
      </Box>
    </Grid.Col>
    <Grid.Col span={6}>
      <Box style={{ border: '1px solid #ddd', padding: '8px', borderRadius: '8px' }}>
        <Text ta="center" fw={500} mb="sm">Output Image</Text>
        {upScaled ? (
          <Image 
            src={upScaled} 
            alt="Result" 
            fit="contain" 
            width="50%" // Set the width to 50% to cut the size in half
            height="auto" // Maintain aspect ratio
          />
        ) : (
          <Text ta="center" c="dimmed">Placeholder for Output</Text>
        )}
      </Box>
    </Grid.Col>
  </Grid>
)}


      {/* Toggle Result Button with Margin */}
      <Button
        className={classes.control}
        size="md"
        radius="xl"
        onClick={() => ChangeShow((show + 1) % 2)}
        mt="10px" // Set the margin-top to 30 pixels
        >
        Toggle Result
      </Button>
    </div>
  );
}