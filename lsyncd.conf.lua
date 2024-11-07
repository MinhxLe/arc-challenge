settings({
	nodaemon = true,
	insist = true,
})
sync({
	default.rsync,
	source = ".",
	target = "root@69.21.145.219:/root/arc",
	rsync = {
		archive = true,
		compress = true,
		rsh = "ssh -p 41500",
	},
	excludeFrom = ".gitignore",
	-- TODO explicitly whitelist what we want to sync over
})
