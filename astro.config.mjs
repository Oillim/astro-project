import { defineConfig } from 'astro/config';
import tailwind from "@astrojs/tailwind";
import alpinejs from "@astrojs/alpinejs";
import mdx from "@astrojs/mdx"
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
// https://astro.build/config
export default defineConfig({
  integrations: [tailwind(), alpinejs(), mdx()],
  output: 'static',
  markdown: {
		remarkPlugins: [remarkMath],
		rehypePlugins: [
			[
				rehypeKatex,
				{
					// Katex plugin options
				}
			]
		]
	}
});