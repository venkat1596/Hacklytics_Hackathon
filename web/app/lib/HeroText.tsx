import { Button, Container, Text, Title } from '@mantine/core';
import classes from './HeroText.module.css';

export function HeroText() {
  return (
    <Container className={classes.wrapper} size={1400}>
      <div className={classes.inner}>
        <Title className={classes.title}>
          Elevate <span className={classes.highlight}>MRI Imaging</span> with AI-Driven <br /> 
          <Text component="span" inherit>
            1.5T to 3T Upgrades
          </Text>
        </Title>

        <Container p={0} size={600}>
          <Text size="lg" c="dimmed" className={classes.description}>
            Transform your MRI capabilities with our innovative AI technology, enhancing image quality, 
            reducing scanning times, and optimizing costs for better patient outcomes. 
            Experience the future of medical imaging today.
          </Text>
        </Container>

        <div className={classes.controls}>
          <Button className={classes.control} size="lg" variant="default" color="gray">
            Request a Demo
          </Button>
          <Button className={classes.control} size="lg">
            Learn More
          </Button>
        </div>
      </div>
    </Container>
  );
}

export default HeroText;
