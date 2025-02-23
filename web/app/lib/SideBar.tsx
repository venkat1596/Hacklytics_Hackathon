'use client';
import { useState, useEffect } from 'react';
import { useRouter, usePathname } from 'next/navigation';
import {
  Icon2fa,
  IconBellRinging,
  IconDatabaseImport,
  IconFingerprint,
  IconKey,
  IconLogout,
  IconReceipt2,
  IconSettings,
  IconSwitchHorizontal,
} from '@tabler/icons-react';
import { Code, Group } from '@mantine/core';
import classes from './NavbarSimpleColored.module.css';

const data = [
  { link: '/', label: 'Overview', icon: IconBellRinging },
  { link: '/application', label: 'Application', icon: IconDatabaseImport },
  { link: '/workflow', label: 'Workflow', icon: IconReceipt2 },
  { link: 'https://github.com/venkat1596/Hacklytics_Hackathon', label: 'Repository', icon: IconFingerprint },
  { link: 'https://hacklytics2025.devpost.com/?preview_token=1Eh9XQPZDc5KLXKykrn1G2vf%2FlgAeiq5c6m0VfYN9i8%3D', label: 'DevPost', icon: IconKey },
];

export function NavbarSimpleColored() {
  const [active, setActive] = useState('Overview');
  const router = useRouter();
  const pathname = usePathname();

  useEffect(() => {
    const currentTab = data.find((item) => item.link === pathname);
    if (currentTab) {
      setActive(currentTab.label);
    }
  }, [pathname]);

  const handleTabClick = (label: string, link: string) => {
    setActive(label);
    if (link.startsWith('http')) {
      window.open(link, '_blank', 'noopener,noreferrer');
    } else if (link) {
      router.push(link);
    }
  };

  const links = data.map((item) => (
    <a
      className={classes.link}
      data-active={item.label === active || undefined}
      href={item.link}
      key={item.label}
      onClick={(event) => {
        event.preventDefault();
        handleTabClick(item.label, item.link);
      }}
    >
      <item.icon className={classes.linkIcon} stroke={1.5} />
      <span>{item.label}</span>
    </a>
  ));

  return (
    <nav className={classes.navbar}>
      <div className={classes.navbarMain}>
        <Group className={classes.header} justify="space-between">
          <Code fw={700} className={classes.version}>
            MIR.AI
          </Code>
        </Group>
        {links}
      </div>
      <div className={classes.footer}>
        <a
          href="https://www.youtube.com/watch?v=dQw4w9WgXcQ"
          className={classes.link}
          onClick={(event) => {
            event.preventDefault();
            window.open("https://www.youtube.com/watch?v=dQw4w9WgXcQ", '_blank', 'noopener,noreferrer');
          }}
        >
          <IconSwitchHorizontal className={classes.linkIcon} stroke={1.5} />
          <span>Demo Video</span>
        </a>

        <a
          href="https://docs.google.com/document/d/1KhXCInHOQa2Ig_ewabaQrM6fq7R-X_bgoTpXDDzRFKM/edit?usp=sharing"
          className={classes.link}
          onClick={(event) => {
            event.preventDefault();
            window.open("https://docs.google.com/document/d/1KhXCInHOQa2Ig_ewabaQrM6fq7R-X_bgoTpXDDzRFKM/edit?usp=sharing", '_blank', 'noopener,noreferrer');
          }}
        >
          <IconLogout className={classes.linkIcon} stroke={1.5} />
          <span>Papers</span>
        </a>
      </div>
    </nav>
  );
}
