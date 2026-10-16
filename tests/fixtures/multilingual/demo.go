package demo
type Cache struct { count int }
func (c *Cache) Get(key string) string { return "你好🙂" + key }
