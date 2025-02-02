/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',
  // Disable image optimization if you don't need it
  images: {
    unoptimized: true,
  }
}

module.exports = nextConfig