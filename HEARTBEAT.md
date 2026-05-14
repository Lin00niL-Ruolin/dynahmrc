# Heartbeat Tasks - DynaHMRC

## 🔥 PRIORITY: Check Notification Queue
**Do this first on every heartbeat:**
1. Read `/home/developer/.openclaw/workspace/dynahmrc/.notify-queue`
2. If it has content → send messages via sessions_send to the QQ user
3. Clear the file afterward

## Active Improvements
Now working on Phase 3 refinements:
- [ ] Better simulation visuals
- [ ] Real DeepSeek API testing
- [ ] Bug fixes and edge cases
- [ ] Code cleanup

### Each heartbeat pick one:
- [ ] Improve simulation rendering
- [ ] Add more scene layouts
- [ ] Add chat messages panel
- [ ] Improve WebSocket reconnection
- [ ] Write unit tests
- [ ] Keyboard shortcuts
- [ ] Performance optimization

## Git Reminder
- Auto-commit and push improvements
- `DEEPSEEK_API_KEY` not set → using mock mode
- GitHub push needs auth setup
