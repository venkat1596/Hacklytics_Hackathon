import { Button, Container, Group, Image, List, ListItem, Text, ThemeIcon, Title } from '@mantine/core';
import classes from './HeroBullets.module.css';

export function HeroBullets() {
  return (
    <Container size="md">
      <div className={classes.inner}>
        <div className={classes.content}>
          <Title className={classes.title}>
            A <span className={classes.highlight}>modern</span> React <br /> components library
          </Title>
          <Text c="dimmed" mt="md">
            Build fully functional accessible web applications faster than ever – Mantine includes
            more than 120 customizable components and hooks to cover you in any situation
          </Text>

          <List mt={30} spacing="sm" size="sm">
            <ListItem>
              <b>TypeScript based</b> – build type-safe applications, all components and hooks
              export types
            </ListItem>
            <ListItem>
              <b>Free and open source</b> – all packages have MIT license, you can use Mantine in
              any project
            </ListItem>
            <ListItem>
              <b>No annoying focus ring</b> – focus ring will appear only when user navigates with
              keyboard
            </ListItem>
          </List>

          <Group mt={30}>
            <Button radius="xl" size="md" className={classes.control}>
              Get started
            </Button>
            <Button variant="default" radius="xl" size="md" className={classes.control}>
              Source code
            </Button>
          </Group>
        </div>
        {/* ✅ Corrected image usage */}
        <Image src="/image.png" alt="Hero image" className={classes.image} />
      </div>
    </Container>
  );
}

export default HeroBullets;
