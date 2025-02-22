'use client';
import Image from 'next/image';
import { ActionIcon, Anchor, Group } from '@mantine/core';
import classes from './FooterCentered.module.css';

const links = [
  { link: '#', label: 'Contact' },
  { link: '#', label: 'Privacy' },
  { link: '#', label: 'Blog' },
  { link: '#', label: 'Store' },
  { link: '#', label: 'Careers' },
];

export function FooterCentered() {
  const items = links.map((link) => (
    <Anchor
      c="dimmed"
      key={link.label}
      href={link.link}
      lh={1}
      onClick={(event) => event.preventDefault()}
      size="sm"
    >
      {link.label}
    </Anchor>
  ));

  return (
    <div className={classes.footer}>
      <div className={classes.inner}>
        <Group className={classes.links}>{items}</Group>

        <Group gap="xs" justify="flex-end" wrap="nowrap">
          <ActionIcon size="lg" variant="default" radius="xl">
            <Image src="/icon--google.svg" alt="Google" width={24} height={24} />
          </ActionIcon>
          <ActionIcon size="lg" variant="default" radius="xl">
            <Image src="/icon--youtube.svg" alt="YouTube" width={24} height={24} />
          </ActionIcon>
          <ActionIcon size="lg" variant="default" radius="xl">
            <Image src="/icon--instagram.svg" alt="Instagram" width={24} height={24} />
          </ActionIcon>
        </Group>
      </div>
    </div>
  );
}
