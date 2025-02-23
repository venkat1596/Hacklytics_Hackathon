import { Container, SimpleGrid, Text, ThemeIcon, Title } from '@mantine/core';
import { IconMicroscope, IconBuildingHospital, IconUserHeart, IconRobot } from '@tabler/icons-react'; // Import icons
import classes from './FeaturesImages.module.css';

const data = [
  {
    title: 'Radiologists',
    description: 'Experience enhanced image clarity and detail, leading to more accurate diagnoses and improved patient care.',
    icon: <IconMicroscope size={30} />, // Icon for Radiologists
  },
  {
    title: 'Healthcare Administrators',
    description: 'Optimize operational efficiency with reduced scanning times and lower costs, maximizing resource allocation.',
    icon: <IconBuildingHospital size={30} />, // Icon for Healthcare Administrators
  },
  {
    title: 'Patients',
    description: 'Benefit from quicker and more accurate diagnoses, leading to faster treatment plans and improved outcomes.',
    icon: <IconUserHeart size={30} />, // Icon for Patients
  },
  {
    title: 'Technologists',
    description: 'Utilize advanced AI tools to streamline workflows and enhance the overall scanning process.',
    icon: <IconRobot size={30} />, // Icon for Technologists
  },
];

export function FeaturesImages() {
  const items = data.map((item, index) => (
    <div className={classes.item} key={index}>
      <ThemeIcon variant="light" className={classes.itemIcon} size={60} radius="md">
        {item.icon} {/* Use the icon from the data object */}
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
      <Text className={classes.supTitle}>Key Benefits</Text>

      <Title className={classes.title} order={2}>
        Upgrading to <span className={classes.highlight}>3T MRI</span> with AI
      </Title>

      <Container size={660} p={0}>
        <Text c="dimmed" className={classes.description}>
          Our AI-driven solution transforms MRI capabilities by enhancing image quality, reducing scanning times, and lowering costs. 
          Embrace the future of medical imaging for better patient outcomes.
        </Text>
      </Container>

      <SimpleGrid cols={{ base: 1, xs: 2 }} spacing={50} mt={30}>
        {items}
      </SimpleGrid>
    </Container>
  );
}

export default FeaturesImages;
