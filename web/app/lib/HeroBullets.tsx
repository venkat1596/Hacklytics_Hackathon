import { Button, Container, Group, Image, List, ListItem, Text, ThemeIcon, Title } from '@mantine/core';
import classes from './HeroBullets.module.css';

export function HeroBullets() {
  return (
    <Container size="md">
      <div className={classes.inner}>
        <div className={classes.content}>
          <Title className={classes.title}>
            Transforming <span className={classes.highlight}>Healthcare</span> with AI-Driven <br /> MRI Super-Resolution
          </Title>
          <Text c="dimmed" mt="md">
            Revolutionize brain pathology diagnosis with advanced MRI super-resolution technology. Our deep learning-powered solution enhances image quality, reduces scanning time, and lowers costs for better patient outcomes.
          </Text>

          <List mt={30} spacing="sm" size="sm">
            <ListItem>
              <b>Deep Learning-Powered</b> – Leverages RFB-ESRGAN and nESRGAN networks for superior texture and frequency restoration in MRI images.
            </ListItem>
            <ListItem>
              <b>Efficient and Cost-Effective</b> – Achieves high-resolution 3D reconstruction with reduced computational costs compared to traditional methods.
            </ListItem>
            <ListItem>
              <b>Innovative Noise Integration</b> – Uses noise-based super-resolution to restore high-frequency details and improve visual clarity.
            </ListItem>
          </List>

          <Group mt={30}>
            <Button radius="xl" size="md" className={classes.control}>
              Learn More
            </Button>
            <Button variant="default" radius="xl" size="md" className={classes.control}>
              Request a Demo
            </Button>
          </Group>
        </div>
        {/* ✅ Kept the same image as before */}
        <Image src="/image.png" alt="AI-Powered MRI Super-Resolution" className={classes.image} />
      </div>
    </Container>
  );
}

export default HeroBullets;