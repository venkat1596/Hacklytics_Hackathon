import { Container, Image, SimpleGrid, Text, ThemeIcon, Title } from '@mantine/core';
import classes from './FeaturesImages.module.css';

const data = [
  {
    title: 'Pharmacists',
    description: 'Azurill can be seen bouncing and playing on its big, rubbery tail',
  },
  {
    title: 'Lawyers',
    description: 'Fans obsess over the particular length and angle of its arms',
  },
  {
    title: 'Bank owners',
    description: 'They divvy up their prey evenly among the members of their pack',
  },
  {
    title: 'Others',
    description: 'Phanpy uses its long nose to shower itself',
  },
];

export function FeaturesImages() {
  const items = data.map((item, index) => (
    <div className={classes.item} key={index}>
      <ThemeIcon variant="light" className={classes.itemIcon} size={60} radius="md">
        <Image src={`${process.env.PUBLIC_URL}/globe.svg`} />
      </ThemeIcon>

      <div>
        <Text fw={700} fz="lg" className={classes.itemTitle}>
          {item.title}
        </Text>
        <Text c="dimmed">{item.description}</Text>
      </div>
    </div>
  ));

  return (
    <Container size={700} className={classes.wrapper}>
      <Text className={classes.supTitle}>Use cases</Text>

      <Title className={classes.title} order={2}>
        PharmLand is <span className={classes.highlight}>not</span> just for pharmacists
      </Title>

      <Container size={660} p={0}>
        <Text c="dimmed" className={classes.description}>
          Its lungs contain an organ that creates electricity. The crackling sound of electricity
          can be heard when it exhales. Azurill’s tail is large and bouncy. It is packed full of the
          nutrients this Pokémon needs to grow.
        </Text>
      </Container>

      <SimpleGrid cols={{ base: 1, xs: 2 }} spacing={50} mt={30}>
        {items}
      </SimpleGrid>
    </Container>
  );
}
