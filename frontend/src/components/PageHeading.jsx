export default function PageHeading({ eyebrow, title, accent, description }) {
  return <header className="page-heading"><span className="eyebrow">{eyebrow}</span><h1>{title}<br/><em>{accent}</em></h1><p>{description}</p></header>;
}
