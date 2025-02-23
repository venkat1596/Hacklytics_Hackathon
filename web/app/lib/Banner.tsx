'use client';
import { Container } from '@mantine/core'; // Removed unused imports
import Image from 'next/image'; // Import Image from next/image
import classes from './Banner.module.css';

export function Banner() {
  return (
    <header className={classes.header}>
      <Container size="md" className={classes.inner}>
        {/* Centered Logo */}
        <Image
          src="/logo.png" // Path to your logo image in the public folder
          alt="Logo"
          width={100} // Set the width of the logo
          height={50} // Set the height of the logo
          className={classes.logo} // Add a class for styling if needed
        />
      </Container>
    </header>
  );
}