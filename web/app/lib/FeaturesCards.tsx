import {
  Badge,
  Card,
  Container,
  Group,
  SimpleGrid,
  Text,
  Title,
} from '@mantine/core';
import classes from './FeaturesCards.module.css';

const teamMembers = [
  {
    name: 'Team Member 1',
    role: 'Machine Learning Developer',
    description: 'Developed the machine learning model that powers our AI-driven MRI solution.',
  },
  {
    name: 'Team Member 2',
    role: 'ML Backend Developer',
    description: 'Built the backend infrastructure for image processing, ensuring efficient data handling.',
  },
  {
    name: 'Team Member 3',
    role: 'Frontend Developer',
    description: 'Created the frontend UI with seamless integration of backend services for optimal user experience.',
  },
  {
    name: 'Team Member 4',
    role: 'ML Parameter Tester',
    description: 'Conducted extensive parameter testing to refine the machine learning models and enhance performance.',
  },
];

export function FeaturesCards() {
  const members = teamMembers.map((member) => (
    <Card key={member.name} shadow="md" radius="md" className={classes.card} padding="xl">
      <img src="/global.svg" alt={member.name} style={{ width: 50, height: 50 }} />
      <Text fz="lg" fw={500} className={classes.cardTitle} mt="md">
        {member.name}
      </Text>
      <Text fz="sm" c="dimmed" mt="sm">
        <b>{member.role}</b> - {member.description}
      </Text>
    </Card>
  ));

  return (
    <Container size="lg" py="xl">
      <Group justify="center">
        <Badge variant="filled" size="lg">
          Meet Our Team
        </Badge>
      </Group>

      <Title order={2} className={classes.title} ta="center" mt="sm">
        The Experts Behind Our AI Solution
      </Title>

      <Text c="dimmed" className={classes.description} ta="center" mt="md">
        Our talented team combines their skills to innovate and enhance MRI imaging technology, ensuring 
        better outcomes for patients and healthcare providers.
      </Text>

      <SimpleGrid cols={{ base: 1, md: 2, lg: 4 }} spacing="xl" mt={50}>
        {members}
      </SimpleGrid>
    </Container>
  );
}

export default FeaturesCards;
