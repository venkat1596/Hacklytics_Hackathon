import { Button, Container, Text, Title } from '@mantine/core';
import classes from './HeroText.module.css';

export function HeroText() {
  return (
    <Container className={classes.wrapper} size={1400}>
      <div className={classes.inner}>
        <Title className={classes.title}>
          High Quality{' '}
          <Text component="span" className={classes.highlight} inherit>
            MRI Scan
          </Text>{' '}
          for any Machine
        </Title>

        <Container p={0} size={600}>
          <Text size="lg" c="dimmed" className={classes.description}>
            Upload an image of the lower-quality scan you would like to upscale.
            Then see the magic in action!
          </Text>
        </Container>

        <div className={classes.controls}>
          <Button
            className={classes.control}
            size="lg"
            variant="default"
            color="gray"
            component="a"
            href="https://www.youtube.com/watch?v=dQw4w9WgXcQ"
            target="_blank"
            rel="noopener noreferrer"
          >
            Demo Video
          </Button>
          <Button
            className={classes.control}
            size="lg"
            component="a"
            href="https://docs.google.com/document/d/your-doc-id/edit"
            target="_blank"
            rel="noopener noreferrer"
          >
            Documentation
          </Button>
        </div>
      </div>
    </Container>
  );
}

export default HeroText;
