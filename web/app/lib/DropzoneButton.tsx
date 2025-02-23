'use client';
import { useRef, useState, useEffect } from 'react';
import { IconCloudUpload, IconDownload, IconX } from '@tabler/icons-react';
import { Button, Group, Text, useMantineTheme, RingProgress } from '@mantine/core';
import { Dropzone } from '@mantine/dropzone';
import classes from './DropzoneButton.module.css';
import axios from 'axios';

export function DropzoneButton() {
  const theme = useMantineTheme();
  const openRef = useRef<() => void>(null);
  const [uploadStatus, setUploadStatus] = useState<'idle' | 'success' | 'error'>('idle');
  const [uploadMessage, setUploadMessage] = useState('');
  const [upScaled,setUpscaled] = useState<any>(null); 
  const [loading, changelstate] = useState<number>(0);
  const [show, ChangeShow] = useState<number>(1);


  // Function to handle file upload
  const handleDrop = async (files: File[]) => {
    console.log("Within Function");
    if (!files.length) return;
    while(loading);
    changelstate(1)
    const file = files[0];
    const data = new FormData();
    data.append('image', file, file.name); // Wrap the file in FormData
  
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
    changelstate(0)
  };
  const ImageLoaderWrapper = () =>
  {
    useEffect(() => {
      ImageLoader()

    }, [loading]) ;
  }

  const ImageLoader = () => {
  
    if (upScaled === null) {
      return null;
    }
  
    if (upScaled === 'loading') {
      
    }
  
    return <img src={upScaled} alt={"Result"} />;
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

      <Button className={classes.control} size="md" radius="xl" onClick={() => ChangeShow((show + 1) % 2)}>
        Toggle Result
      </Button>
      
      {loading ? (
    <RingProgress 
      sections={[
        { value: 60, color: 'cyan' },
        { value: 20, color: 'pink' }
      ]} 
    />
      ) : upScaled !== null && show === 0? (
     <img src={upScaled} alt="Result" />
     ) : null}
    </div>
  );
}
