import { Button, Container, Group, Image, List, ListItem, Text, Title } from '@mantine/core';
import Link from 'next/link'; // Import Link from next/link
import classes from './HeroBullets.module.css';

export function HeroBullets() {
  return (
    <Container size="md">
      <div className={classes.inner}>
        <div className={classes.content}>
          <Title className={classes.title}>
            Enhancing <span className={classes.highlight}>Healthcare</span> through AI-Driven <br /> MRI Technology Upgrade
          </Title>
          <Text c="dimmed" mt="md">
            Unlock the potential of advanced MRI systems with our AI-powered solutions that facilitate the transition from 1.5T to 3T. Experience superior image quality, reduced scanning times, and lower operational costs, leading to improved patient outcomes and diagnostic accuracy.
          </Text>

          <List mt={30} spacing="sm" size="sm">
            <ListItem>
              <b>Superior Image Quality</b> – AI algorithms enhance image clarity and detail, enabling precise diagnoses with high-resolution imaging.
            </ListItem>
            <ListItem>
              <b>Increased Efficiency</b> – Streamlined workflows and automated processes reduce scanning times, allowing healthcare providers to serve more patients efficiently.
            </ListItem>
            <ListItem>
              <b>Predictive Analytics</b> – Leverage AI to analyze vast datasets for informed decision-making and personalized patient care, improving overall treatment outcomes.
            </ListItem>
          </List>

          <Group mt={30}>
            {/* Features Button with Link */}
            <Link href="/about" passHref legacyBehavior>
              <Button
                component="a" // Render the button as an anchor tag
                radius="xl"
                size="md"
                className={classes.control}
              >
                Features
              </Button>
            </Link>

            {/* Demo Video Button with Link */}
            <Link href="/demo-video" passHref legacyBehavior>
              <Button
                component="a" // Render the button as an anchor tag
                variant="default"
                radius="xl"
                size="md"
                className={classes.control}
              >
                Demo Video
              </Button>
            </Link>
          </Group>
        </div>
        {/* ✅ Kept the same image as before */}
        <Image src="/image.png" alt="AI-Driven MRI Technology Upgrade" className={classes.image} />
      </div>
    </Container>
  );
}

export default HeroBullets;