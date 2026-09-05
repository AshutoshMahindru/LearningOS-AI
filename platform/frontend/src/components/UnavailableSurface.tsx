import { EmptyState } from './EmptyState';

type UnavailableSurfaceProps = {
  title: string;
};

export function UnavailableSurface({ title }: UnavailableSurfaceProps) {
  return (
    <EmptyState
      title={title}
      message="This surface is not available yet. The shell keeps a generic route so later work can land without mission-specific UI."
    />
  );
}
