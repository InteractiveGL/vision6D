import React from "react";
// import styles from "./index.module.css";
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
    <Link className="" to={props.to}>
      <div className="">{props.icon}</div>
      <div className="">{props.title}</div>
      <p className="">{props.description}</p>
    </Link>
  );
};

export default Card;
