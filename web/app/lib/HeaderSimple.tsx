'use client';
import { useState } from 'react';
import { Burger, Container, Group } from '@mantine/core';
import { useDisclosure } from '@mantine/hooks';
import Link from 'next/link'; // Import Link from next/link
import Image from 'next/image'; // Import Image from next/image
import classes from './HeaderSimple.module.css';
import Logo from '../../public/Logo.png'

const links = [
  { link: '/', label: 'Home' },
  { link: '/about', label: 'Features' },
  { link: '/team', label: 'Team' },
];

export function HeaderSimple() {
  const [opened, { toggle }] = useDisclosure(false);
  const [active, setActive] = useState(links[0].link);

  const items = links.map((link) => (
    <Link // Use Link from next/link
      key={link.label}
      href={link.link} // Use href prop for navigation
      className={classes.link}
    >
      <span onClick={() => setActive(link.link)}>{link.label}</span>
    </Link>
  ));

  return (
    <header className={classes.header}>
      <Container size="md" className={classes.inner}>
        {/* Add the Image component here */}
        <Image
          src={Logo} /* Path to your logo image in the public folder */
          alt="Logos"
          width={100}
          height={50} 
          className={classes.logo} 
        />

        <Group gap={5} visibleFrom="xs">
          {items}
        </Group>

        <Burger opened={opened} onClick={toggle} hiddenFrom="xs" size="sm" />
      </Container>
    </header>
  );
}