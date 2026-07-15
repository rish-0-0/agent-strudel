const patterns = [
  'sound("bd sd [bd sd] sd")',
  'sound("[bd ~ bd bd] [~ sd] [bd ~] [sd sd bd ~]")',
  'sound("bd(3,8) sd(2,8) hh(8,8)").slow(2)',
  'note("c2 eb2 g2 <c3 bb2>").sound("sawtooth").slow(2)',
  'sound("bd sd hh sd").jux(rev)',
  'sound("bd(5,8), <sd cp> hh(7,8)").slow(1.5)',
];

export async function generate(prompt = "") {
  return patterns[Math.floor(Math.random() * patterns.length)];
}
