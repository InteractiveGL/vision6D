import React from "react";
import Link from "@docusaurus/Link";

interface CardProp {
  title: string;
  description: string;
  to: string;
  icon: string;
}

const Card: React.FC<CardProp> = (props: CardProp) => {
  return (
    // <Link className={styles.card} to={props.to}>
    //   <div className={styles.cardIcon}>{props.icon}</div>
    //   <div className={styles.cardTitle}>{props.title}</div>
    //   <p className={styles.cardDesc}>{props.description}</p>
    // </Link>
    <Link
      className="
      bg-[var(--ifm-background-surface-color)]
      border border-[var(--ifm-color-emphasis-200)]
      rounded-[10px]
      p-6
      no-underline text-inherit
      transition-transform transition-shadow duration-200 ease-in-out
      flex flex-col justify-between
      shadow-[0_1px_4px_rgba(0,0,0,0.04)]
      hover:-translate-y-[3px]
      hover:shadow-[0_6px_16px_rgba(0,0,0,0.08)]"
      to={props.to}
    >
      <div className="text-[2rem] mb-3">{props.icon}</div>
      <div className="text-[1.15rem] font-semibold mb-2">{props.title}</div>
      <p className="text-[0.95rem] text-[var(--ifm-color-content-secondary)]">
        {props.description}
      </p>
    </Link>
  );
};

export default Card;
