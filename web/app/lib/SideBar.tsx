'use client';
import { useState, useEffect } from 'react';
import { useRouter, usePathname } from 'next/navigation'; // Import useRouter and usePathname
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
  { link: '/resources', label: 'Resources', icon: IconReceipt2 }, // Add route for Resources
  { link: 'https://github.com/venkat1596/Hacklytics_Hackathon', label: 'Repository', icon: IconFingerprint },
  { link: 'https://hacklytics2025.devpost.com/?preview_token=1Eh9XQPZDc5KLXKykrn1G2vf%2FlgAeiq5c6m0VfYN9i8%3D', label: 'DevPost', icon: IconKey },
];

export function NavbarSimpleColored() {
  const [active, setActive] = useState('Overview'); // Default active tab
  const router = useRouter(); // Initialize useRouter
  const pathname = usePathname(); // Get current pathname

  // Sync active state with current route
  useEffect(() => {
    const currentTab = data.find((item) => item.link === pathname);
    if (currentTab) {
      setActive(currentTab.label);
    }
  }, [pathname]);

  const handleTabClick = (label: string, link: string) => {
    setActive(label);
    if (link.startsWith('http')) {
      // External link (e.g., GitHub, DevPost)
      window.location.href = link;
    } else if (link) {
      // Internal route (e.g., /resources)
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
            v3.01.01
          </Code>
        </Group>
        {links}
      </div>
      <div className={classes.footer}>
        <a href="#" className={classes.link} onClick={(event) => event.preventDefault()}>
          <IconSwitchHorizontal className={classes.linkIcon} stroke={1.5} />
          <span>Demo Video</span>
        </a>

        <a href="#" className={classes.link} onClick={(event) => event.preventDefault()}>
          <IconLogout className={classes.linkIcon} stroke={1.5} />
          <span>Papers</span>
        </a>
      </div>
    </nav>
  );
}