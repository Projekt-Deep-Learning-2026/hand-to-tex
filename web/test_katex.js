import katex from 'katex';
const html = katex.renderToString("a^2 + b^2 = c^2", { displayMode: true, output: 'mathml' });
console.log(html);