// page.tsx
import Image from "next/image";
import { HeaderSimple } from "./lib/HeaderSimple"; // Adjust the import path as necessary
import { HeroBullets } from "./lib/HeroBullets"; // Adjust the import path as necessary
import { FooterCentered } from "./lib/FooterCentered"; // Adjust the import path as necessary
import styles from "./page.module.css";

export default function Home() {
  return (
    <div className="home">
      <HeaderSimple />
      <HeroBullets />
      <FooterCentered />
    </div>
  );
}

