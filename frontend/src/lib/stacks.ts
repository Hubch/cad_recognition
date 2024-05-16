// Keep in sync with backend (prompts/types.py)
export enum Stack {
  HTML_TAILWIND = "html_tailwind",
  REACT_TAILWIND = "react_tailwind",
  BOOTSTRAP = "bootstrap",
  VUE_TAILWIND = "vue_tailwind",
  IONIC_TAILWIND = "ionic_tailwind",
  SVG = "svg",
  JSON = "json",
  PDF = "pdf",
  JPG = 'jpg',
}

export const STACK_DESCRIPTIONS: {
  [key in Stack]: { components: string[]; inBeta: boolean };
} = {
  svg: { components: ["SVG"], inBeta: true },
  json: { components: ["JSON"], inBeta: true },
  pdf: { components: ["PDF"], inBeta: true },
  jpg: { components: ["JPG"], inBeta: true },
  html_tailwind: { components: ["HTML", "Tailwind"], inBeta: false },
  react_tailwind: { components: ["React", "Tailwind"], inBeta: false },
  bootstrap: { components: ["Bootstrap"], inBeta: false },
  vue_tailwind: { components: ["Vue", "Tailwind"], inBeta: true },
  ionic_tailwind: { components: ["Ionic", "Tailwind"], inBeta: true },
};
