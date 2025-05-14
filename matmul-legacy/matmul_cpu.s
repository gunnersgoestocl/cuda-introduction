	.arch armv8-a
	.file	"matmul_cpu.c"
	.text
	.align	2
	.global	initMatrix
	.type	initMatrix, %function
initMatrix:
.LFB6:
	.cfi_startproc
	sub	sp, sp, #48
	.cfi_def_cfa_offset 48
	str	x0, [sp, 24]
	str	d0, [sp, 16]
	str	w1, [sp, 12]
	str	wzr, [sp, 44]
	b	.L2
.L5:
	str	wzr, [sp, 40]
	b	.L3
.L4:
	ldr	w1, [sp, 44]
	ldr	w0, [sp, 12]
	mul	w1, w1, w0
	ldr	w0, [sp, 40]
	add	w0, w1, w0
	sxtw	x0, w0
	lsl	x0, x0, 3
	ldr	x1, [sp, 24]
	add	x0, x1, x0
	ldr	d0, [sp, 16]
	str	d0, [x0]
	ldr	w0, [sp, 40]
	add	w0, w0, 1
	str	w0, [sp, 40]
.L3:
	ldr	w1, [sp, 40]
	ldr	w0, [sp, 12]
	cmp	w1, w0
	blt	.L4
	ldr	w0, [sp, 44]
	add	w0, w0, 1
	str	w0, [sp, 44]
.L2:
	ldr	w1, [sp, 44]
	ldr	w0, [sp, 12]
	cmp	w1, w0
	blt	.L5
	nop
	nop
	add	sp, sp, 48
	.cfi_def_cfa_offset 0
	ret
	.cfi_endproc
.LFE6:
	.size	initMatrix, .-initMatrix
	.align	2
	.global	matMulCpu
	.type	matMulCpu, %function
matMulCpu:
.LFB7:
	.cfi_startproc
	sub	sp, sp, #48
	.cfi_def_cfa_offset 48
	str	x0, [sp, 24]
	str	x1, [sp, 16]
	str	x2, [sp, 8]
	str	w3, [sp, 4]
	str	wzr, [sp, 44]
	b	.L7
.L12:
	str	wzr, [sp, 40]
	b	.L8
.L11:
	str	wzr, [sp, 36]
	b	.L9
.L10:
	ldr	w1, [sp, 44]
	ldr	w0, [sp, 4]
	mul	w1, w1, w0
	ldr	w0, [sp, 40]
	add	w0, w1, w0
	sxtw	x0, w0
	lsl	x0, x0, 3
	ldr	x1, [sp, 8]
	add	x0, x1, x0
	ldr	d1, [x0]
	ldr	w1, [sp, 44]
	ldr	w0, [sp, 4]
	mul	w1, w1, w0
	ldr	w0, [sp, 36]
	add	w0, w1, w0
	sxtw	x0, w0
	lsl	x0, x0, 3
	ldr	x1, [sp, 24]
	add	x0, x1, x0
	ldr	d2, [x0]
	ldr	w1, [sp, 36]
	ldr	w0, [sp, 4]
	mul	w1, w1, w0
	ldr	w0, [sp, 40]
	add	w0, w1, w0
	sxtw	x0, w0
	lsl	x0, x0, 3
	ldr	x1, [sp, 16]
	add	x0, x1, x0
	ldr	d0, [x0]
	fmul	d0, d2, d0
	ldr	w1, [sp, 44]
	ldr	w0, [sp, 4]
	mul	w1, w1, w0
	ldr	w0, [sp, 40]
	add	w0, w1, w0
	sxtw	x0, w0
	lsl	x0, x0, 3
	ldr	x1, [sp, 8]
	add	x0, x1, x0
	fadd	d0, d1, d0
	str	d0, [x0]
	ldr	w0, [sp, 36]
	add	w0, w0, 1
	str	w0, [sp, 36]
.L9:
	ldr	w1, [sp, 36]
	ldr	w0, [sp, 4]
	cmp	w1, w0
	blt	.L10
	ldr	w0, [sp, 40]
	add	w0, w0, 1
	str	w0, [sp, 40]
.L8:
	ldr	w1, [sp, 40]
	ldr	w0, [sp, 4]
	cmp	w1, w0
	blt	.L11
	ldr	w0, [sp, 44]
	add	w0, w0, 1
	str	w0, [sp, 44]
.L7:
	ldr	w1, [sp, 44]
	ldr	w0, [sp, 4]
	cmp	w1, w0
	blt	.L12
	nop
	nop
	add	sp, sp, 48
	.cfi_def_cfa_offset 0
	ret
	.cfi_endproc
.LFE7:
	.size	matMulCpu, .-matMulCpu
	.align	2
	.global	calculateElapsedTime
	.type	calculateElapsedTime, %function
calculateElapsedTime:
.LFB8:
	.cfi_startproc
	sub	sp, sp, #32
	.cfi_def_cfa_offset 32
	stp	x0, x1, [sp, 16]
	stp	x2, x3, [sp]
	ldr	x1, [sp]
	ldr	x0, [sp, 16]
	sub	x0, x1, x0
	fmov	d0, x0
	scvtf	d1, d0
	ldr	x1, [sp, 8]
	ldr	x0, [sp, 24]
	sub	x0, x1, x0
	fmov	d0, x0
	scvtf	d0, d0
	adrp	x0, .LC0
	ldr	d2, [x0, #:lo12:.LC0]
	fmul	d0, d0, d2
	fadd	d0, d1, d0
	add	sp, sp, 32
	.cfi_def_cfa_offset 0
	ret
	.cfi_endproc
.LFE8:
	.size	calculateElapsedTime, .-calculateElapsedTime
	.align	2
	.global	terminate
	.type	terminate, %function
terminate:
.LFB9:
	.cfi_startproc
	stp	x29, x30, [sp, -32]!
	.cfi_def_cfa_offset 32
	.cfi_offset 29, -32
	.cfi_offset 30, -24
	mov	x29, sp
	str	x0, [sp, 24]
	ldr	x0, [sp, 24]
	bl	perror
	mov	w0, 1
	bl	exit
	.cfi_endproc
.LFE9:
	.size	terminate, .-terminate
	.section	.rodata
	.align	3
.LC1:
	.string	"%f "
	.text
	.align	2
	.global	debugMatrix
	.type	debugMatrix, %function
debugMatrix:
.LFB10:
	.cfi_startproc
	stp	x29, x30, [sp, -48]!
	.cfi_def_cfa_offset 48
	.cfi_offset 29, -48
	.cfi_offset 30, -40
	mov	x29, sp
	str	x0, [sp, 24]
	str	w1, [sp, 20]
	str	wzr, [sp, 44]
	b	.L17
.L20:
	str	wzr, [sp, 40]
	b	.L18
.L19:
	ldr	w1, [sp, 44]
	ldr	w0, [sp, 20]
	mul	w1, w1, w0
	ldr	w0, [sp, 40]
	add	w0, w1, w0
	sxtw	x0, w0
	lsl	x0, x0, 3
	ldr	x1, [sp, 24]
	add	x0, x1, x0
	ldr	d0, [x0]
	adrp	x0, .LC1
	add	x0, x0, :lo12:.LC1
	bl	printf
	ldr	w0, [sp, 40]
	add	w0, w0, 1
	str	w0, [sp, 40]
.L18:
	ldr	w1, [sp, 40]
	ldr	w0, [sp, 20]
	cmp	w1, w0
	blt	.L19
	mov	w0, 10
	bl	putchar
	ldr	w0, [sp, 44]
	add	w0, w0, 1
	str	w0, [sp, 44]
.L17:
	ldr	w1, [sp, 44]
	ldr	w0, [sp, 20]
	cmp	w1, w0
	blt	.L20
	nop
	nop
	ldp	x29, x30, [sp], 48
	.cfi_restore 30
	.cfi_restore 29
	.cfi_def_cfa_offset 0
	ret
	.cfi_endproc
.LFE10:
	.size	debugMatrix, .-debugMatrix
	.section	.rodata
	.align	3
.LC2:
	.string	"Usage matmul_cpu dim_size"
	.align	3
.LC3:
	.string	"elapsed time %f\n"
	.text
	.align	2
	.global	main
	.type	main, %function
main:
.LFB11:
	.cfi_startproc
	stp	x29, x30, [sp, -96]!
	.cfi_def_cfa_offset 96
	.cfi_offset 29, -96
	.cfi_offset 30, -88
	mov	x29, sp
	str	w0, [sp, 28]
	str	x1, [sp, 16]
	ldr	w0, [sp, 28]
	cmp	w0, 1
	bgt	.L22
	adrp	x0, .LC2
	add	x0, x0, :lo12:.LC2
	bl	terminate
.L22:
	ldr	x0, [sp, 16]
	add	x0, x0, 8
	ldr	x0, [x0]
	bl	atoi
	str	w0, [sp, 92]
	ldrsw	x1, [sp, 92]
	ldrsw	x0, [sp, 92]
	mul	x0, x1, x0
	lsl	x0, x0, 3
	bl	malloc
	str	x0, [sp, 80]
	ldrsw	x1, [sp, 92]
	ldrsw	x0, [sp, 92]
	mul	x0, x1, x0
	lsl	x0, x0, 3
	bl	malloc
	str	x0, [sp, 72]
	ldrsw	x1, [sp, 92]
	ldrsw	x0, [sp, 92]
	mul	x0, x1, x0
	lsl	x0, x0, 3
	bl	malloc
	str	x0, [sp, 64]
	ldr	w1, [sp, 92]
	fmov	d0, 3.0e+0
	ldr	x0, [sp, 80]
	bl	initMatrix
	ldr	w1, [sp, 92]
	adrp	x0, .LC4
	ldr	d0, [x0, #:lo12:.LC4]
	ldr	x0, [sp, 72]
	bl	initMatrix
	ldr	w1, [sp, 92]
	movi	d0, #0
	ldr	x0, [sp, 64]
	bl	initMatrix
	add	x0, sp, 48
	mov	x1, x0
	mov	w0, 0
	bl	clock_gettime
	ldr	w3, [sp, 92]
	ldr	x2, [sp, 64]
	ldr	x1, [sp, 72]
	ldr	x0, [sp, 80]
	bl	matMulCpu
	add	x0, sp, 32
	mov	x1, x0
	mov	w0, 0
	bl	clock_gettime
	ldp	x2, x3, [sp, 32]
	ldp	x0, x1, [sp, 48]
	bl	calculateElapsedTime
	adrp	x0, .LC3
	add	x0, x0, :lo12:.LC3
	bl	printf
	mov	w0, 0
	ldp	x29, x30, [sp], 96
	.cfi_restore 30
	.cfi_restore 29
	.cfi_def_cfa_offset 0
	ret
	.cfi_endproc
.LFE11:
	.size	main, .-main
	.section	.rodata
	.align	3
.LC0:
	.word	-400107883
	.word	1041313291
	.align	3
.LC4:
	.word	-1717986918
	.word	1069128089
	.ident	"GCC: (GNU) 11.4.1 20231218 (Red Hat 11.4.1-3)"
	.section	.note.GNU-stack,"",@progbits
