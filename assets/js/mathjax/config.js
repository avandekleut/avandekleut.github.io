
// MathJax = {
//   tex: {
//     inlineMath: [['$', '$']]
//   },
//   svg: {
//     fontCache: 'global'
//   },
//   options: {
//     renderActions: {
//       /* add a new named action not to override the original 'find' action */
//       find_script_mathtex: [10, function (doc) {
//         for (const node of document.querySelectorAll('script[type^="math/tex"]')) {
//           const display = !!node.type.match(/; *mode=display/);
//           const math = new doc.options.MathItem(node.textContent, doc.inputJax[0], display);
//           const text = document.createTextNode('');
//           node.parentNode.replaceChild(text, node);
//           math.start = {node: text, delim: '', n: 0};
//           math.end = {node: text, delim: '', n: 0};
//           doc.math.push(math);
//         }
//       }, '']
//     }
//   }
// };


MathJax = {
  tex: {
    inlineMath: [['$', '$']],
    displayMath: [['$$', '$$']]
  },
  svg: {
    fontCache: 'global'
  }
};
//
// (function () {
//   var script = document.createElement('script');
//   script.src = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js';
//   script.async = true;
//   document.head.appendChild(script);
// })();
//
// MathJax = {
//   options: {
//     renderActions: {
//       /* add a new named action not to override the original 'find' action */
//       find_script_mathtex: [10, function (doc) {
//         for (const node of document.querySelectorAll('script[type^="math/tex"]')) {
//           const display = !!node.type.match(/; *mode=display/);
//           const math = new doc.options.MathItem(node.textContent, doc.inputJax[0], display);
//           const text = document.createTextNode('');
//           node.parentNode.replaceChild(text, node);
//           math.start = {node: text, delim: '', n: 0};
//           math.end = {node: text, delim: '', n: 0};
//           doc.math.push(math);
//         }
//       }, '']
//     }
//   }
// };
