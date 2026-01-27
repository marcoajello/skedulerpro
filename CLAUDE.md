# SkedulerPro Website

## Site Info
- **Domain:** www.skedulerpro.com
- **Hosting:** GitHub Pages
- **Repo:** github.com/marcoajello/skedulerpro

## Downloads
- **Releases repo:** github.com/marcoajello/skeduler_beta/releases
- **Current version:** 1.4.0-beta
- **Files:**
  - `Skeduler-{version}-arm64.dmg` (Apple Silicon)
  - `Skeduler-{version}.dmg` (Intel)

## Updating Beta Version
1. Upload new .dmg files to GitHub Releases (via github.com, not Desktop)
2. Update download links in `index.html`
3. Update version in the beta-label span
4. Commit and push via GitHub Desktop

## File Structure
- `index.html` - Main landing page with downloads and tutorial videos
- `confirmation.html` - Email confirmation success page
- `Skeduler_web_LOGO.png` - Logo image
- `CNAME` - Custom domain config

## Design
- Dark theme: #1a1a1a background
- Accent color: #d32f2f (red)
- Font: System fonts (-apple-system, etc.)
